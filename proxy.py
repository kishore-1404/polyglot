"""proxy.py — Unified reverse proxy for split compute architecture.

Routes:
- /llama/{path} -> http://127.0.0.1:LLAMA_PORT/{path}
- /tts/{path}   -> http://127.0.0.1:TTS_PORT/{path}
"""
import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Load config to get local ports dynamically
try:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # If running with custom ports, load them, otherwise fallback to defaults
    LLAMA_PORT = cfg.get("llama_server", {}).get("port", 8088)
    TTS_PORT = cfg.get("supertonic", {}).get("port", 9988)
except Exception:
    LLAMA_PORT = 8088
    TTS_PORT = 9988

LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
TTS_URL = f"http://127.0.0.1:{TTS_PORT}"

app = FastAPI(title="Polyglot Split Compute Proxy")
client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


@app.api_route("/llama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_llama(request: Request, path: str):
    target_url = f"{LLAMA_URL}/{path}"
    return await _do_proxy(request, target_url)


@app.api_route("/tts/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_tts(request: Request, path: str):
    target_url = f"{TTS_URL}/{path}"
    return await _do_proxy(request, target_url)


async def _do_proxy(request: Request, target_url: str):
    # Forward headers (excluding host)
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    
    # Read request body
    body = await request.body()
    
    # Send request to target server with streaming support
    req = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
        params=request.query_params
    )
    
    resp = await client.send(req, stream=True)
    
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=dict(resp.headers)
    )
