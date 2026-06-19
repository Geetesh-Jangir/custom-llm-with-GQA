# 🚀 Custom LLM with Grouped Query Attention (GQA)

A modern decoder-only Large Language Model built from scratch in PyTorch featuring **Grouped Query Attention (GQA)**, **KV Cache**, **RoPE**, **RMSNorm**, **SwiGLU**, **Sliding Window Attention**, and **Supervised Fine-Tuning (SFT)**.

This project demonstrates how modern LLM architectures such as LLaMA, Mistral, and Gemma can be implemented from first principles while remaining lightweight, modular, and easy to understand.

---

## ✨ Features

### Modern Transformer Architecture

* Decoder-only GPT-style architecture
* Multi-Head Self Attention
* Grouped Query Attention (GQA)
* Rotary Positional Embeddings (RoPE)
* RMSNorm
* SwiGLU Feed Forward Network
* Residual Connections
* Causal Masking

### Efficient Inference

* KV Cache for autoregressive generation
* Sliding Window Attention
* Attention Sink Tokens
* Reduced memory footprint through GQA
* Faster token generation through cached attention states

### Training Optimizations

* Mixed Precision Training (FP16)
* Gradient Accumulation
* Learning Rate Scheduling
* Checkpointing
* TensorBoard Logging
* Custom Tokenization Pipeline

### Fine-Tuning

* Supervised Fine-Tuning (SFT)
* Curriculum Learning
* Instruction Dataset Formatting
* Custom Data Collators

---

# 🏗️ Architecture

```text
Input Tokens
      │
      ▼
Token Embeddings
      │
      ▼
┌─────────────────────────┐
│ Transformer Block × N   │
│                         │
│ RMSNorm                │
│ Grouped Query Attention│
│ RoPE                   │
│ KV Cache              │
│ Residual Connection    │
│ RMSNorm               │
│ SwiGLU FFN            │
│ Residual Connection    │
└─────────────────────────┘
      │
      ▼
Final RMSNorm
      │
      ▼
Linear Head
      │
      ▼
Next Token Prediction
```

---

# 🧠 Core Technologies

## 1. Grouped Query Attention (GQA)

Traditional Multi-Head Attention creates separate Query, Key, and Value projections for every head.

GQA reduces memory usage by allowing multiple query heads to share fewer key-value heads.

### Standard Multi-Head Attention

```text
8 Query Heads
8 Key Heads
8 Value Heads
```

### Grouped Query Attention

```text
8 Query Heads
2 Key Heads
2 Value Heads
```

Multiple query heads reuse the same KV representations.

### Benefits

✅ Lower memory consumption

✅ Smaller KV Cache

✅ Faster inference

✅ Better scalability for long-context generation

---

## 2. KV Cache

During autoregressive generation, previously computed keys and values are stored.

Without KV Cache:

```text
Token 1 → compute
Token 1,2 → recompute
Token 1,2,3 → recompute
...
```

With KV Cache:

```text
Store K,V once
Reuse previous K,V
Compute only newest token
```

### Benefits

* Avoids repeated attention computations
* Significantly improves generation speed
* Essential for production LLM inference

---

## 3. Rotary Positional Embeddings (RoPE)

The model uses Rotary Position Embeddings instead of learned positional embeddings.

Advantages:

* Better extrapolation to longer contexts
* Improved attention behavior
* Used in LLaMA, Mistral, Gemma, Qwen, DeepSeek

---

## 4. RMSNorm

Instead of LayerNorm:

```python
y = x / RMS(x)
```

Benefits:

* Faster computation
* Lower overhead
* More stable training

---

## 5. SwiGLU

Modern feed-forward network:

```text
FFN
 └── SwiGLU
```

Benefits:

* Better parameter efficiency
* Stronger representations
* Used in LLaMA-family models

---

## 6. Sliding Window Attention

Only attends to a fixed recent context window.

Benefits:

* Memory efficient
* Enables long-context generation
* Reduces quadratic attention cost

---

## 7. Attention Sink

Special preserved tokens remain accessible even when the attention window shifts.

Benefits:

* Better long-context retention
* More stable generation

---

# 📁 Project Structure

```text
custom-llm-with-GQA/
│
├── base-model/
│   ├── model_modern.py
│   ├── block_modern.py
│   ├── attn_modern.py
│   ├── kv_cache.py
│   ├── rope_custom.py
│   ├── rmsnorm.py
│   ├── swiglu.py
│   ├── tokenizer.py
│   ├── tokenizer_bpe.py
│   ├── dataset_bpe.py
│   ├── checkpointing.py
│   ├── lr_scheduler.py
│   ├── demo_generate.py
│   ├── sample.py
│   └── train.py
│
├── fine-tuning/
│   ├── train_sft.py
│   ├── dataset_sft.py
│   ├── curriculum.py
│   ├── collator_sft.py
│   ├── formatters.py
│   └── sample_sft.py
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Geetesh-Jangir/custom-llm-with-GQA.git

cd custom-llm-with-GQA
```

## Create Environment

```bash
python -m venv venv

source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install tokenizers datasets tensorboard tqdm
```

---

# 🚀 Training Base Model

Example:

```bash
python train.py \
    --data tiny.txt \
    --out runs/part-demo \
    --bpe \
    --vocab_size 8000 \
    --epochs 1 \
    --steps 500 \
    --batch_size 16 \
    --block_size 128 \
    --n_layer 2 \
    --n_head 2 \
    --n_embd 128
```

---

# 🎯 Text Generation

```bash
python sample.py \
    --ckpt runs/part-demo/model_last.pt \
    --tokens 100 \
    --prompt "Generate a short story"
```

---

# ⚡ KV Cache Demo

```bash
python demo_generate.py \
    --tokens 500 \
    --rope \
    --rmsnorm \
    --swiglu
```

The script compares:

```text
Generation with KV Cache
vs
Generation without KV Cache
```

allowing direct inference benchmarking.

---

# 🎓 Supervised Fine-Tuning

Train instruction-following behavior:

```bash
python train_sft.py \
    --steps 200 \
    --batch_size 8 \
    --block_size 256
```

Features:

* Instruction tuning
* Prompt-response training
* Curriculum learning
* Custom batching pipeline

---

# 📊 Model Components

| Component                | Implemented |
| ------------------------ | ----------- |
| Decoder-only Transformer | ✅           |
| GQA                      | ✅           |
| KV Cache                 | ✅           |
| RoPE                     | ✅           |
| RMSNorm                  | ✅           |
| SwiGLU                   | ✅           |
| Sliding Window Attention | ✅           |
| Attention Sink           | ✅           |
| Mixed Precision          | ✅           |
| Gradient Accumulation    | ✅           |
| Curriculum Learning      | ✅           |
| SFT                      | ✅           |

---

# 📊 Benchmark Results

Benchmarks were conducted on an NVIDIA RTX 3050 Laptop GPU using dedicated benchmarking scripts included in the repository.

## KV Cache Performance

| Metric            | Without KV Cache | With KV Cache |
| ----------------- | ---------------- | ------------- |
| Generation Time   | 3.534 s          | 0.475 s       |
| Tokens / Second   | 141.48           | 1051.77       |
| Inference Speedup | 1.0×             | **7.43×**     |

### Key Takeaway

By storing previously computed Key-Value states and reusing them during autoregressive decoding, KV Cache reduced generation latency by **86.6%** and increased throughput by more than **7×**.

---

## GQA Memory Efficiency

| Metric           | Standard MHA | GQA       |
| ---------------- | ------------ | --------- |
| Generation Time  | 3.899 s      | 3.697 s   |
| Peak GPU Memory  | 0.263 GB     | 0.239 GB  |
| Tokens / Second  | 128.23       | 135.24    |
| Memory Reduction | —            | **8.83%** |

### Key Takeaway

Grouped Query Attention reduced KV memory requirements while slightly improving throughput, demonstrating a more efficient attention mechanism for inference workloads.

---

## Long Context Performance

| Context Length | Generation Time | Tokens/sec | Peak Memory |
| -------------- | --------------- | ---------- | ----------- |
| 128            | 2.703 s         | 73.98      | 0.11 GB     |
| 256            | 0.718 s         | 278.68     | 0.11 GB     |
| 512            | 2.364 s         | 84.59      | 0.12 GB     |
| 1024           | 0.497 s         | 402.05     | 0.14 GB     |
| 2048           | 1.603 s         | 124.78     | 0.18 GB     |

### Key Takeaway

The model successfully handled context windows up to 2048 tokens while maintaining a relatively small GPU memory footprint through the combination of GQA, KV Cache, and sliding-window attention mechanisms.

---

## Experimental GQA Benchmark

A dedicated benchmark comparing Multi-Head Attention and Grouped Query Attention showed a significant reduction in generation time:

| Architecture | Time    |
| ------------ | ------- |
| Standard MHA | 1.903 s |
| GQA          | 0.062 s |

Observed Speedup: **30.76×**

Note: This benchmark was conducted on a lightweight experimental configuration and primarily highlights the computational advantages of reduced Key-Value projections. Real-world improvements depend on model size, sequence length, hardware, and workload characteristics.


# 🔬 Learning Objectives

This project demonstrates:

* Building GPT-style models from scratch
* Implementing modern transformer optimizations
* Understanding GQA internals
* KV Cache engineering
* Efficient inference systems
* RoPE mathematics
* Modern normalization techniques
* LLM fine-tuning workflows

---

# 🛠 Future Improvements

* Flash Attention
* LoRA Fine-Tuning
* Quantization (INT8 / 4-bit)
* Distributed Training
* Multi-GPU Support
* Retrieval-Augmented Generation (RAG)
* RLHF / DPO Training
* Long Context (>32K Tokens)

---

# 👨‍💻 Author

**Geetesh Jangir**

BCA Graduate | Machine Learning Engineer | LLM Enthusiast

GitHub: https://github.com/Geetesh-Jangir

---

# ⭐ If you found this project useful

Give the repository a star and feel free to contribute.
