"""
Interactive Terminal Chat with Qwen 3.5 4B (Q4_K_M GGUF)
Optimized for 13th Gen Intel Core i5 (CPU Inference)
"""

import os
import sys
from llama_cpp import Llama

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen_Qwen3.5-4B-Q4_K_M.gguf")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at: {MODEL_PATH}")
        print("Please run `python download_model.py` first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" Loading Qwen 3.5 4B (Q4_K_M Quantized) ...")
    print(" Running natively on CPU with AVX2 & multi-threading")
    print("=" * 60)

    # Initialize model
    # n_threads=8 matches the 8 physical cores on i5-13420H for best throughput
    # n_ctx=4096 gives ample context while keeping RAM usage around ~3.5GB total
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_batch=512,
        verbose=False
    )

    print("\nModel loaded successfully! Type '/exit' or '/quit' to quit, '/clear' to reset chat history.\n")

    system_prompt = "You are Qwen 3.5, a helpful, intelligent, and accurate AI assistant."
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break

            if user_input.lower() in ["/clear", "clear"]:
                conversation_history = [{"role": "system", "content": system_prompt}]
                print("Conversation history cleared.")
                continue

            conversation_history.append({"role": "user", "content": user_input})

            print("\nQwen: ", end="", flush=True)

            # Stream response
            stream = llm.create_chat_completion(
                messages=conversation_history,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stream=True
            )

            assistant_reply = ""
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    assistant_reply += content

            print()
            conversation_history.append({"role": "assistant", "content": assistant_reply})

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error during generation: {e}]")

if __name__ == "__main__":
    main()

