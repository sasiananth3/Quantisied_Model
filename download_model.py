"""
Reliable Downloader for Qwen 3.5 4B Q4_K_M GGUF
- Direct chunked streaming download with live progress bar
- No deprecated warnings or symlink issues
- Automatic SHA-256 integrity verification
"""

import os
import sys
import time
import hashlib
import requests

URL = "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
FILENAME = "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
EXPECTED_SIZE = 3013027808
EXPECTED_SHA256 = "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PATH = os.path.join(CURRENT_DIR, FILENAME)
TEMP_PATH = DEST_PATH + ".download"

def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} TB"

def main():
    print("=" * 65)
    print(f"Model: {FILENAME}")
    print(f"Target Size: {format_size(EXPECTED_SIZE)} ({EXPECTED_SIZE:,} bytes)")
    print(f"Destination: {DEST_PATH}")
    print("=" * 65, flush=True)

    # Check if already downloaded and valid
    if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) == EXPECTED_SIZE:
        print("\nModel file already exists. Checking SHA-256 integrity...", flush=True)
        h = hashlib.sha256()
        with open(DEST_PATH, "rb") as f:
            while chunk := f.read(16 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest().lower() == EXPECTED_SHA256.lower():
            print("Integrity check passed! Model is ready to use.\n", flush=True)
            return
        else:
            print("Existing file is incomplete/corrupted. Re-downloading...", flush=True)
            os.remove(DEST_PATH)

    # Start download
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    print("\nConnecting to Hugging Face CDN...", flush=True)
    response = requests.get(URL, headers=headers, stream=True, timeout=30)

    if response.status_code != 200:
        print(f"Error: Failed to connect (HTTP {response.status_code})", file=sys.stderr)
        sys.exit(1)

    print("Connected! Downloading with live progress...\n", flush=True)

    chunk_size = 4 * 1024 * 1024  # 4 MB chunks
    downloaded = 0
    start_time = time.time()
    last_print = start_time

    sha_calc = hashlib.sha256()

    with open(TEMP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                sha_calc.update(chunk)
                downloaded += len(chunk)
                now = time.time()

                if now - last_print >= 1.0 or downloaded == EXPECTED_SIZE:
                    elapsed = now - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    pct = (downloaded / EXPECTED_SIZE) * 100
                    remaining = (EXPECTED_SIZE - downloaded) / speed if speed > 0 else 0

                    bar_len = 30
                    filled = int(bar_len * downloaded / EXPECTED_SIZE)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    print(
                        f"\r[{bar}] {pct:5.1f}% | {format_size(downloaded)}/{format_size(EXPECTED_SIZE)} "
                        f"| {format_size(speed)}/s | ETA: {int(remaining // 60):02d}m {int(remaining % 60):02d}s",
                        end="",
                        flush=True
                    )
                    last_print = now

    print("\n\nVerifying downloaded file SHA-256...", flush=True)
    if sha_calc.hexdigest().lower() == EXPECTED_SHA256.lower():
        print("Checksum verified successfully! (SHA-256 match)", flush=True)
    else:
        print("Warning: Checksum mismatch. The file may be incomplete.", file=sys.stderr)

    if os.path.exists(DEST_PATH):
        os.remove(DEST_PATH)
    os.rename(TEMP_PATH, DEST_PATH)

    total_time = time.time() - start_time
    print(f"\nDownload finished in {int(total_time // 60)}m {int(total_time % 60)}s!")
    print(f"Model saved to: {DEST_PATH}")
    print("\nYou can now run:\n  python chat.py\n  python test_inference.py\n", flush=True)

if __name__ == "__main__":
    main()
