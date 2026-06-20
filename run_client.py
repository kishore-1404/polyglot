#!/usr/bin/env python3
"""run_client.py — Launch the Polyglot Web UI client.

Run this script on your local/interaction laptop.
It will verify your configuration and spin up the main pipeline and Web UI.
"""
import os
import sys
import yaml
import subprocess


def main():
    print("=" * 60)
    print("🌍 POLYGLOT CLIENT INTERACTION RUNNER 🌍")
    print("=" * 60)

    # 1. Check config
    if not os.path.exists("config.yaml"):
        print("❌ Error: config.yaml not found!")
        print("   Please copy config.yaml to the current directory.")
        sys.exit(1)

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    llama_url = cfg.get("llama_server", {}).get("url")
    tts_url = cfg.get("supertonic", {}).get("url")

    if not llama_url or not tts_url:
        print("\n⚠️ Warning: No remote URLs configured in config.yaml.")
        print("   If you want to run compute locally, make sure llama-server and supertonic are running.")
        print("   If you want to use a remote GPU host, run 'python3 run_backend.py' on that host,")
        print("   and paste the generated URLs into your config.yaml.")
    else:
        print(f"\n🔗 Configured Remote Endpoints:")
        print(f"   Llama-Server: {llama_url}")
        print(f"   Supertonic:   {tts_url}")

    print("\n🚀 Starting client web UI...")
    try:
        # Launch main.py with --web
        cmd = [sys.executable, "main.py", "--web"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching client: {e}")


if __name__ == "__main__":
    main()
