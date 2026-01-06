# WIGIP v2

# WIGIP v2: 1.7B ViT-Text Model

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Download%20Weights-yellow)](https://huggingface.co/Nottybro/Wigip_v2_1.7B_ViT)

> **📥 Download Model Weights:**
> The pre-trained JAX/Flax checkpoints (Stage 1) are hosted on Hugging Face.
> [**Click here to download the weights**](https://huggingface.co/Nottybro/Wigip_v2_1.7B_ViT)

---

## Stage 1 – Text Pre-Training (ViT-Style Transformer)

WIGIP-1 v2 is an experimental research model exploring **Vision Transformer (ViT) style architectures for text modeling**, implemented using **JAX + Flax** with **Fully Sharded Data Parallelism (FSDP)** via `pjit`.

This repository currently contains **ONLY Phase 1 (Text Pre-Training)**.

---

## ⚠️ Training Status (IMPORTANT)

- ✅ **Phase 1: Text-only pre-training**
  - Character-level language modeling
  - Dataset: **C4 (English)**
  - Architecture: ViT-style transformer applied to reshaped text
  - **~57,000 training steps completed**
  - Training performed using streaming data and FSDP

- ❌ **Phase 2: Image training (NOT DONE)**
  - No image data has been used
  - No multimodal or vision supervision yet
  - This phase is planned for future work

🚨 **Model weights will be updated in the future once Phase 2 training is performed.**
Do NOT treat current checkpoints as a final or multimodal-capable model.

---

## 🧠 Model Overview (Phase 1)

- Text is tokenized at **character level**
- Tokens are reshaped into a **2D grid**
- Grid is treated like an image and processed using:
  - Patch embedding via convolution
  - Multi-head self-attention
  - Feed-forward blocks
- Final output predicts the **next character token**

This phase is intended to test whether **ViT-style inductive biases** can learn meaningful structure from text alone.

---

## ⚙️ Technical Highlights

- JAX + Flax + Optax
- `pjit` with 2D mesh (`data`, `model`)
- Activation rematerialization (`nn.remat`)
- Gradient clipping
- Warmup + cosine learning rate schedule
- Streaming dataset (no full dataset in memory)

---

## 💾 Checkpointing

- Checkpoints are:
  - Automatically saved at time intervals
  - Compressed into `.zip` archives
  - Contain:
    - Model parameters (`.pkl.gz`)
    - Optimizer state
    - Training step metadata
- Training can be safely resumed from the latest zipped checkpoint

---

## 🔮 Future Work

- Phase 2: Image-based training
- Multimodal alignment (text + vision)
- Scaling beyond current step count
- Improved tokenization strategies
- Evaluation on downstream tasks

---

## ⚠️ Disclaimer

This is **research code** and an **experimental architecture**.
Results are preliminary and **not production-ready**.

