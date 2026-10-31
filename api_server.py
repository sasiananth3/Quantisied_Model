"""
OpenAI-Compatible Local API Server for Qwen 3.5 4B (Q4_K_M GGUF)
Enables integration with Open WebUI, LM Studio, Cline, Continue, etc.
Endpoint: http://127.0.0.1:8000/v1
"""

import os
import sys
import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen_Qwen3.5-4B-Q4_K_M.gguf")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run `python download_model.py` first.")
        sys.exit(1)

    print("=" * 65)
    print(" Starting OpenAI-Compatible API Server for Qwen 3.5 4B ...")
    print(f" Model: {MODEL_PATH}")
    print(" Server URL: http://127.0.0.1:8000/v1")
    print(" Compatible with Open WebUI, Cursor, Cline, LangChain, etc.")
    print("=" * 65)

    server_settings = ServerSettings(
        host="127.0.0.1",
        port=8000
    )

    model_settings = [
        ModelSettings(
            model=MODEL_PATH,
            model_alias="qwen-3.5-4b",
            n_ctx=4096,
            n_threads=8,
            n_batch=512
        )
    ]

    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings
    )

    uvicorn.run(app, host=server_settings.host, port=server_settings.port)

if __name__ == "__main__":
    main()

