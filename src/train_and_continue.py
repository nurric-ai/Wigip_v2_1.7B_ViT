# === WIGIP-1 v2: STAGE 1 - TEXT PRE-TRAINING (with Automatic Zipping) ===
!pip install datasets
import jax, jax.numpy as jnp, flax.linen as nn, optax, numpy as np, pickle, os, time, glob, gzip, shutil
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit
from flax.training import train_state
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
from jax import lax
from flax.core import freeze, unfreeze

# --- 1. CONFIGURATION ---
# Model Parameters for the full ~1.7B model
N_LAYER = 24
N_EMBD = 2496 # The full 1.7B parameter model
N_HEAD = 32
# ViT/Image Configuration
IMG_SIZE = 180 
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
BLOCK_SIZE = IMG_SIZE * IMG_SIZE # 256
# Training
HOURS_TO_RUN = 6.0 
STEPS_PER_HOUR_ESTIMATE = 9500 
GRADIENT_ACCUMULATION_STEPS = 8
BATCH_SIZE = 2
TOTAL_TRAIN_STEPS = 610000 
SAVE_INTERVAL_HOURS = 8.0
LEARNING_RATE = 1e-4; WARMUP_STEPS = 2000
PARAM_DTYPE = jnp.bfloat16; COMPUTE_DTYPE = jnp.bfloat16
CHECKPOINT_DIR = '/kaggle/working/wigip1_v2_checkpoints' 

# --- 2. JAX SHARDING SETUP ---
print("--- Setting up 2D device mesh for FSDP... ---")
num_devices = jax.device_count()
assert num_devices % 2 == 0
device_mesh = np.array(jax.devices()).reshape((2, num_devices // 2))
mesh = Mesh(device_mesh, axis_names=('data', 'model'))
print(f"Device mesh created with shape: {device_mesh.shape}")
fsdp_sharding = PartitionSpec(None, 'model'); data_sharding = PartitionSpec('data', None)

# --- 3. DATA & VOCAB ---
print("\n--- Loading dataset and building vocabulary... ---");
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
processed_text_sample = ""
for i, example in enumerate(dataset.take(1000)): processed_text_sample += example['text']
chars = sorted(list(set(processed_text_sample))); vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}; encode = lambda s: [stoi.get(c, 0) for c in s]
print(f"Vocabulary size: {vocab_size}")

# --- 4. ViT-Style MODEL DEFINITION ---
class PatchEmbedding(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=N_EMBD, kernel_size=(PATCH_SIZE, PATCH_SIZE), strides=(PATCH_SIZE, PATCH_SIZE), padding='VALID', dtype=COMPUTE_DTYPE)(x)
        return x.reshape(x.shape[0], -1, N_EMBD)
class AttentionBlock(nn.Module):
    n_head: int; n_embd: int
    @nn.compact
    def __call__(self, x):
        x = x + nn.SelfAttention(num_heads=self.n_head, qkv_features=self.n_embd, dtype=COMPUTE_DTYPE)(nn.LayerNorm(dtype=COMPUTE_DTYPE)(x))
        ffn_out = nn.Sequential([
            nn.Dense(4 * self.n_embd, dtype=COMPUTE_DTYPE), nn.gelu, nn.Dense(self.n_embd, dtype=COMPUTE_DTYPE)
        ])(nn.LayerNorm(dtype=COMPUTE_DTYPE)(x))
        x = x + ffn_out
        return x
class ViTStyleTransformer(nn.Module):
    @nn.compact
    def __call__(self, idx):
        tok_emb = nn.Embed(num_embeddings=vocab_size, features=PATCH_SIZE*PATCH_SIZE, dtype=COMPUTE_DTYPE)(idx)
        x = tok_emb.reshape(tok_emb.shape[0], IMG_SIZE, IMG_SIZE, -1)
        x = PatchEmbedding()(x)
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, NUM_PATCHES, N_EMBD))
        x = x + pos_emb.astype(COMPUTE_DTYPE)
        for _ in range(N_LAYER):
            x = nn.remat(AttentionBlock)(n_head=N_HEAD, n_embd=N_EMBD)(x)
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE)(x)
        logits = nn.Dense(features=vocab_size, dtype=jnp.float32)(x[:, 0])
        return logits

# --- 5. TRAINING LOGIC ---
def get_stream_batches(streaming_dataset):
    buffer = [];
    for example in streaming_dataset:
        tokens=encode(example['text']); buffer.extend(tokens)
        while len(buffer)>=(BATCH_SIZE*BLOCK_SIZE)+1:
            all_chunks=np.array(buffer[:BATCH_SIZE*BLOCK_SIZE+1]); x=all_chunks[:-1].reshape(BATCH_SIZE,BLOCK_SIZE); 
            y = all_chunks[1:].reshape(BATCH_SIZE, BLOCK_SIZE)[:, -1:]
            buffer=buffer[BATCH_SIZE*BLOCK_SIZE:]; yield x,y
class TrainState(train_state.TrainState):pass

@partial(pjit, static_argnames='total_steps', in_shardings=None, out_shardings=NamedSharding(mesh,PartitionSpec()))
def create_train_state(total_steps):
    rng=jax.random.PRNGKey(0); model=ViTStyleTransformer(); 
    data_shape=(BATCH_SIZE // mesh.shape['data'], BLOCK_SIZE); 
    params=model.init(rng,jnp.zeros(data_shape,dtype=jnp.int32))['params']
    param_count=sum(p.size for p in jax.tree_util.tree_leaves(params)); print(f"Model parameter count: {param_count/1e6:.2f}M")
    lr_schedule=optax.warmup_cosine_decay_schedule(init_value=0.0,peak_value=LEARNING_RATE,warmup_steps=WARMUP_STEPS,decay_steps=total_steps-WARMUP_STEPS,end_value=LEARNING_RATE/10)
    tx=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr_schedule))
    return TrainState.create(apply_fn=model.apply,params=params,tx=tx)

@partial(pjit,in_shardings=(NamedSharding(mesh,PartitionSpec()),NamedSharding(mesh,data_sharding),NamedSharding(mesh,data_sharding)),out_shardings=(NamedSharding(mesh,PartitionSpec()),NamedSharding(mesh,PartitionSpec())))
def train_step(state,x,y):
    def loss_fn(params):
        logits=state.apply_fn({'params':params},x.astype(jnp.int32)); 
        y_squeezed = jnp.squeeze(y.astype(jnp.int32))
        return optax.softmax_cross_entropy_with_integer_labels(logits,y_squeezed).mean()
    loss,grads=jax.value_and_grad(loss_fn)(state.params);
    state = state.apply_gradients(grads=grads)
    return state,loss

# --- NEW: ZIPPING AND UNZIPPING LOGIC ---
def save_checkpoint_zipped(state, step):
    folder_name = f"step_{step}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, folder_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    with gzip.open(os.path.join(checkpoint_path, "state_and_opt.pkl.gz"), 'wb') as f:
        pickle.dump({'step': state.step, 'opt_state': jax.device_get(state.opt_state)}, f)
        
    params_path = os.path.join(checkpoint_path, "params")
    os.makedirs(params_path, exist_ok=True)
    unfrozen_params = unfreeze(state.params)
    for key, value in unfrozen_params.items():
        with gzip.open(os.path.join(params_path, f"{key}.pkl.gz"), 'wb') as f:
            pickle.dump(jax.device_get(value), f)

    # --- ZIP THE FOLDER ---
    zip_path = os.path.join(CHECKPOINT_DIR, f"step_{step}")
    shutil.make_archive(zip_path, 'zip', checkpoint_path)
    # --- CLEAN UP THE ORIGINAL FOLDER ---
    shutil.rmtree(checkpoint_path)
    print(f"Zipped checkpoint saved to {zip_path}.zip")

def load_checkpoint_zipped(state_shell):
    # Find the latest zipped checkpoint
    zip_files = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.zip"))
    if not zip_files: return None, 0
    latest_zip = max(zip_files, key=os.path.getctime)
    
    print(f"Found latest checkpoint: {latest_zip}. Unzipping...")
    unzip_dir = os.path.join(CHECKPOINT_DIR, "temp_unzip")
    if os.path.exists(unzip_dir): shutil.rmtree(unzip_dir)
    shutil.unpack_archive(latest_zip, unzip_dir, 'zip')

    latest_dir = unzip_dir
    print(f"Resuming training from {latest_dir}...")
    
    with gzip.open(os.path.join(latest_dir, "state_and_opt.pkl.gz"), 'rb') as f:
        meta_data = pickle.load(f)
    params_path = os.path.join(latest_dir, "params")
    loaded_params = {}
    for key in state_shell.params.keys():
        with gzip.open(os.path.join(params_path, f"{key}.pkl.gz"), 'rb') as f:
            loaded_params[key] = pickle.load(f)
            
    state = state_shell.replace(params=freeze(loaded_params), opt_state=meta_data['opt_state'], step=meta_data['step'])
    start_step = int(state.step)
    
    # Clean up the temporary unzipped folder
    shutil.rmtree(unzip_dir)
    print(f"Successfully reassembled state. Resuming from step {start_step}.")
    return state, start_step

# --- 6. RUN TRAINING ---
with mesh:
    # --- UPGRADED RESUME LOGIC ---
    # When resuming, upload your .zip file to CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    state = create_train_state(TOTAL_TRAIN_STEPS)
    state, start_step = load_checkpoint_zipped(state)
    if state is None:
        state = create_train_state(TOTAL_TRAIN_STEPS)
        start_step = 0
        print("No checkpoint found. Starting new training run from step 0.")

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"\n--- Model Initialized ---"); print(f"Total parameter count: {param_count/1e6:.2f}M")
    batch_generator = get_stream_batches(dataset.shuffle(buffer_size=10000, seed=int(time.time())))
    last_save_time = time.time()
    session_start_time = time.time()
    
    steps_to_run_this_session = int(HOURS_TO_RUN * STEPS_PER_HOUR_ESTIMATE)
    target_step_for_progress_bar = start_step + steps_to_run_this_session
    
    print(f"\n--- Starting ViT-style FSDP training for ~{HOURS_TO_RUN} hours... ---")
    print("NOTE: The first step will be VERY slow due to JIT compilation if starting from scratch.")
    
    try:
        progress_bar = tqdm(desc="Training Steps", initial=start_step, total=target_step_for_progress_bar)
        step = start_step
        while True:
            if (time.time() - session_start_time) / 3600 >= HOURS_TO_RUN:
                print(f"\n{HOURS_TO_RUN} hour time limit reached. Stopping training.")
                break
            
            if step >= TOTAL_TRAIN_STEPS:
                print("\nOverall training goal reached. Stopping training.")
                break

            x, y = next(batch_generator)
            state, loss = train_step(state, x, y)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{float(loss):.4f}")

            current_time = time.time()
            if (current_time - last_save_time) / 3600 >= SAVE_INTERVAL_HOURS:
                print(f"\n{SAVE_INTERVAL_HOURS} hours have passed. Saving checkpoint at step {step+1}...")
                save_checkpoint_zipped(state, step + 1)
                last_save_time = current_time
            
            step += 1
            
    finally:
        print(f"\nExecution finished or interrupted. Performing final save at step {state.step}...")
        save_checkpoint_zipped(state, int(state.step))

