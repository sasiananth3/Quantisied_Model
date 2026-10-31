# Qwen 3.5 4B Quantization Guide

This document explains the technical architecture, quantization schemes, and how to build custom quantizations from raw Hugging Face weights.

---

## 1. Overview of Qwen 3.5 4B

- **Base Model**: `Qwen/Qwen3.5-4B`
- **Architecture**: Hybrid architecture combining Gated DeltaNet linear attention with standard attention layers and native vision-language multimodal capabilities.
- **Vocabulary Size**: ~248k tokens
- **Full Precision Size (FP16/BF16)**: ~8.66 GB
- **Quantized Size (Q4_K_M)**: ~2.80 GB (over **67% memory reduction** with virtually identical output quality)

---

## 2. Quantization Schemes Explained

| Quantization Type | File Size | Memory (RAM/VRAM) | Quality / Perplexity | Recommended Use Case |
|---|---|---|---|---|
| **Q4_K_M** (Recommended) | ~2.80 GB | ~3.5 GB | Near-lossless for 4-bit | Balanced speed and quality on modern CPUs |
| **Q5_K_M** | ~3.20 GB | ~4.0 GB | Higher precision | Good middle-ground if 4-bit shows minor drift |
| **Q8_0** | ~4.30 GB | ~5.2 GB | Identical to FP16 | Benchmarking & maximum precision |
| **IQ3_M** | ~2.31 GB | ~2.8 GB | Good (with imatrix) | Extreme low-RAM environments |

> **What is `K_M`?**  
> `K` refers to k-quants in llama.cpp, which quantize critical attention layers (like `attn_v` and `feed_forward.w2`) at slightly higher precision while compressing standard feed-forward tensors to 4-bit. `M` denotes the Medium variant, providing the best quality-to-size trade-off.

---

## 3. How to Build GGUF Quantization From Scratch (Step-by-Step)

If you ever want to quantize a new fine-tuned checkpoint or raw weights yourself:

### Step 1: Clone `llama.cpp`
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

### Step 2: Download the Raw Weights
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3.5-4B', local_dir='./qwen3.5-4b-raw')"
```

### Step 3: Convert Hugging Face Safetensors to GGUF (BF16)
```bash
python convert_hf_to_gguf.py ./qwen3.5-4b-raw --outfile ./qwen3.5-4b-bf16.gguf --outtype bf16
```

### Step 4: Quantize to Target Precision (e.g. Q4_K_M)
Using `llama-quantize` (compiled binary):
```bash
llama-quantize ./qwen3.5-4b-bf16.gguf ./Qwen_Qwen3.5-4B-Q4_K_M.gguf Q4_K_M
```
*(Optional)* For even higher accuracy, you can generate an importance matrix (`imatrix`) using a calibration dataset before running `llama-quantize`.

---

## 4. Running the Model

### Option A: Interactive Terminal Chat
```bash
python chat.py
```

### Option B: Local OpenAI-Compatible API Server
```bash
python api_server.py
```
Then configure your client (Open WebUI, Continue, Cursor, or custom script):
- **Base URL**: `http://127.0.0.1:8000/v1`
- **Model Name**: `qwen-3.5-4b`
- **API Key**: `not-needed`

