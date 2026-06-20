#!/usr/bin/env python3
"""run_backend.py — Launch and tunnel GPU compute servers.

Run this script on your GPU-enabled host machine.
It will spin up llama-server and supertonic in the background,
launch proxy.py on port 7860, and tunnel it via ngrok.
"""
import os
import sys
import time
import yaml
import subprocess
import socket
import getpass
from pyngrok import ngrok, exception as ngrok_exceptions


def kill_port(port):
    try:
        # Cross-platform way to kill process on port
        if os.name == 'posix':
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            subprocess.run(["pkill", "-f", f"port {port}"], capture_output=True)
    except Exception:
        pass


def wait_for_port(port, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(('127.0.0.1', port))
            s.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    print("=" * 60)
    print("🌍 POLYGLOT COMPUTE HOST RUNNER 🌍")
    print("=" * 60)

    # 1. Load config
    if not os.path.exists("config.yaml"):
        print("❌ Error: config.yaml not found in current directory!")
        sys.exit(1)

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    llm_path = cfg["models"]["llm_gguf"]
    mmproj_path = cfg["models"]["mmproj_gguf"]
    llama_port = cfg.get("llama_server", {}).get("port", 8088)
    tts_port = cfg.get("supertonic", {}).get("port", 9988)
    proxy_port = 7860

    print(f"LLM Model Path:  {llm_path}")
    print(f"Projector Path:  {mmproj_path}")
    print(f"Llama Port:      {llama_port}")
    print(f"TTS Port:        {tts_port}")
    print(f"Proxy Port:      {proxy_port}")

    # 2. Cleanup existing processes
    print("\n🧹 Cleaning up old backend processes...")
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "supertonic"], capture_output=True)
    subprocess.run(["pkill", "-f", "proxy:app"], capture_output=True)
    subprocess.run(["pkill", "-f", "ngrok"], capture_output=True)
    for p in [llama_port, tts_port, proxy_port]:
        kill_port(p)
    time.sleep(2)

    # Make logs dir
    os.makedirs("logs", exist_ok=True)

    # 3. Start llama-server
    print("\n🚀 Starting llama-server...")
    thinking_args = []
    if not cfg.get("llama_server", {}).get("enable_thinking", False):
        thinking_args = ["--chat-template-kwargs", '{"enable_thinking":false}']

    llama_cmd = [
        "llama-server",
        "-m", llm_path,
        "--mmproj", mmproj_path,
        "--host", "0.0.0.0",
        "--port", str(llama_port),
        "-ngl", str(cfg.get("llama_server", {}).get("gpu_layers", 99)),
        "-c", str(cfg.get("llama_server", {}).get("ctx_size", 32768)),
        "--flash-attn", "on",
        "--no-mmap"
    ] + thinking_args

    llama_log = open("logs/llama_server_backend.log", "w")
    llama_proc = subprocess.Popen(llama_cmd, stdout=llama_log, stderr=subprocess.STDOUT, text=True)

    # 4. Start supertonic
    print("🔊 Starting Supertonic V3 serve...")
    tts_cmd = ["supertonic", "serve", "--port", str(tts_port)]
    tts_log = open("logs/supertonic_backend.log", "w")
    tts_proc = subprocess.Popen(tts_cmd, stdout=tts_log, stderr=subprocess.STDOUT, text=True)

    # 5. Wait for backends to wake up
    print("⏳ Waiting for backend servers to spin up...")
    if not wait_for_port(llama_port, timeout=90):
        print("❌ Error: llama-server failed to start in time. Check logs/llama_server_backend.log")
        sys.exit(1)
    print("✅ llama-server is online.")

    if not wait_for_port(tts_port, timeout=45):
        print("❌ Error: Supertonic V3 failed to start in time. Check logs/supertonic_backend.log")
        sys.exit(1)
    print("✅ Supertonic V3 is online.")

    # 6. Start the unified proxy
    print("\n🔀 Starting unified reverse proxy on port 7860...")
    proxy_cmd = [
        sys.executable, "-m", "uvicorn", "proxy:app",
        "--host", "127.0.0.1", "--port", str(proxy_port),
        "--log-level", "warning"
    ]
    proxy_log = open("logs/proxy_backend.log", "w")
    proxy_proc = subprocess.Popen(proxy_cmd, stdout=proxy_log, stderr=subprocess.STDOUT)

    if not wait_for_port(proxy_port, timeout=15):
        print("❌ Error: proxy failed to start in time. Check logs/proxy_backend.log")
        sys.exit(1)
    print("✅ Proxy is online.")

    # 7. Start ngrok tunnel
    print("\n🌐 Establishing ngrok tunnel...")
    tunnel = None
    try:
        tunnel = ngrok.connect(str(proxy_port), "http")
    except ngrok_exceptions.PyngrokNgrokError as e:
        if any(word in str(e).lower() for word in ["authtoken", "authentication", "token"]):
            print("🔑 ngrok authtoken required.")
            token = getpass.getpass("Enter your ngrok Authtoken: ")
            ngrok.set_auth_token(token)
            tunnel = ngrok.connect(str(proxy_port), "http")
        else:
            raise e

    public_url = tunnel.public_url.rstrip('/')
    print("\n" + "=" * 60)
    print("🎉 BACKEND SERVERS ARE ACTIVE AND EXPOSED!")
    print(f"👉 Public Proxy URL: {public_url}")
    print("=" * 60)

    print("\n📋 Copy and paste the following config into the 'config.yaml'")
    print("   on your client/interaction laptop:")
    print("-" * 60)
    print("llama_server:")
    print(f"  url: \"{public_url}/llama\"")
    print("\nsupertonic:")
    print(f"  url: \"{public_url}/tts\"")
    print("-" * 60)

    print("\nKeep this script running. Press Ctrl+C to terminate all servers.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🧹 Shutting down all servers...")
    finally:
        # Terminate processes
        for proc in [llama_proc, tts_proc, proxy_proc]:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Close logs
        llama_log.close()
        tts_log.close()
        proxy_log.close()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
