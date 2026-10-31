"""
Quick verification script for Qwen 3.5 4B Q4_K_M
"""

import os
import sys
import time
from llama_cpp import Llama

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen_Qwen3.5-4B-Q4_K_M.gguf")

def test():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading {os.path.basename(MODEL_PATH)} ...")
    start_load = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        verbose=False
    )
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    prompt = "Explain quantization in artificial intelligence in two simple sentences."
    print(f"\nPrompt: {prompt}\n")

    start_gen = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    gen_time = time.time() - start_gen

    output = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    tok_per_sec = tokens / gen_time if gen_time > 0 else 0

    print("Response:")
    print("-" * 50)
    print(output)
    print("-" * 50)
    print(f"Generated {tokens} tokens in {gen_time:.2f}s (~{tok_per_sec:.1f} tokens/sec)")

if __name__ == "__main__":
    test()

