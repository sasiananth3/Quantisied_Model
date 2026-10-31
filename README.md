# Qwen 3.5 4B (Quantized)

## Definition

The **Quantized Qwen 3.5 4B** is a compressed version of Alibaba's open-source Qwen 3.5 4B model converted into the **GGUF (Q4_K_M)** format. 

By converting the model's weights from 16-bit precision (FP16/BF16) down to 4-bit integers, the file size and RAM requirement are reduced from **~8.7 GB to ~2.8 GB** (a 67% reduction) while preserving over 98% of the original model's intelligence and conversational quality.

---

## Use Cases

- **Local CPU Inference**: Run an intelligent LLM smoothly on everyday laptops and PCs without needing an expensive dedicated GPU.
- **Private & Offline Assistant**: Handle sensitive code, documents, and personal queries with zero data leaving your machine.
- **Local Tool Backend**: Connect the model as an OpenAI-compatible backend (`http://localhost:8000/v1`) to tools like Open WebUI, LM Studio, Cursor, or Cline.
- **Low-Latency Prototyping**: Fast startup and lightweight footprint for testing prompts, RAG pipelines, and agent workflows locally.

---

## Setup Instructions

### 1. Download Model Weights
Run the download script to fetch the quantized model (~2.8 GB) with auto-resume:
```powershell
python download_model.py
```

### 2. Run the Model

Choose how you want to interact with the model:

- **Interactive Terminal Chat**:
  ```powershell
  python chat.py
  ```

- **Quick Verification Test**:
  ```powershell
  python test_inference.py
  ```

- **OpenAI-Compatible Local API Server** (Port 8000):
  ```powershell
  python api_server.py
  ```
