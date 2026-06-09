#!/usr/bin/env python3
"""main.py — Polyglot Live entry point."""
import argparse, sys, time
import requests, yaml
from core.pipeline import Pipeline


def load_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_servers(cfg):
    ok = True
    checks = [
        ("llama-server",  "GET",  f"http://localhost:{cfg['llama_server']['port']}/health", None),
        ("Supertonic V3", "POST", f"http://localhost:{cfg['supertonic']['port']}/v1/tts",
         {"text": "ok", "lang": "en", "voice": "F1"}),
    ]
    print("\n  Server health check:")
    for name, method, url, body in checks:
        try:
            fn = requests.post if method == "POST" else requests.get
            r  = fn(url, json=body, timeout=5)
            badge = "✅" if r.ok else "⚠️ "
            if not r.ok: ok = False
        except Exception:
            badge = "❌"; ok = False
        print(f"    {badge}  {name}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web",    action="store_true")
    parser.add_argument("--term",   action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    print("━" * 54)
    print("  🌍  POLYGLOT LIVE — Multilingual Voice Companion")
    print("━" * 54)

    if not check_servers(cfg):
        print("\n  ⚠️  Run: ./start_servers.sh\n")
        sys.exit(1)
    print("\n  ✅  Servers ready.\n")

    on_state_cbs = []; on_turn_cbs = []
    on_text_ready_cbs = []; on_log_cbs = []
    on_latency_update_cbs = []

    if args.web:
        from ui.web import (run_server, set_control_callback, set_demo_callback,
                            on_state as ws, on_turn as wt,
                            on_text_ready as wtr, on_log as wl,
                            on_latency_update as wlu,
                            on_demo_state as wds)
        port = cfg["ui"]["web_port"]
        run_server(port)
        time.sleep(1.5)
        print(f"  🌐  Web UI → http://localhost:{port}\n")
        on_state_cbs.append(ws); on_turn_cbs.append(wt)
        on_text_ready_cbs.append(wtr); on_log_cbs.append(wl)
        on_latency_update_cbs.append(wlu)

    use_term = (not args.web) or args.term
    term_ui  = None
    if use_term:
        from ui.terminal import TerminalUI
        term_ui = TerminalUI(cfg)
        on_state_cbs.append(term_ui.on_state)
        on_turn_cbs.append(term_ui.on_turn)
        on_text_ready_cbs.append(term_ui.on_text_ready)
        on_log_cbs.append(term_ui.on_log)
        on_latency_update_cbs.append(term_ui.on_latency_update)

    pipeline = Pipeline(
        cfg,
        on_state          = lambda s: [cb(s) for cb in on_state_cbs],
        on_turn           = lambda t: [cb(t) for cb in on_turn_cbs],
        on_text_ready     = lambda t: [cb(t) for cb in on_text_ready_cbs],
        on_log            = lambda m: [cb(m) for cb in on_log_cbs],
        on_latency_update = lambda l: [cb(l) for cb in on_latency_update_cbs],
    )

    # Wire mute control and demo control from web UI → pipeline
    if args.web:
        set_control_callback(pipeline.handle_control)
        set_demo_callback(pipeline.handle_demo_control)
        pipeline.on_demo_state = wds


    try:
        if term_ui:
            term_ui.run(pipeline.run)
        else:
            print("  Pipeline running. Open browser at the URL above.")
            print("  Ctrl+C to quit.\n")
            pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("\n👋  Polyglot stopped.")


if __name__ == "__main__":
    main()
