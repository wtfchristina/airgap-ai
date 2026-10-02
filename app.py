import os
import sys
import glob
import subprocess
import multiprocessing
import platform
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Airgap AI")

LLAMA_SERVER_PORT = 8081
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
llama_process = None

def get_system_hardware():
    threads = multiprocessing.cpu_count()
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return threads, is_apple_silicon

def find_model():
    candidates = glob.glob("models/*.gguf")
    return candidates[0] if candidates else None

def launch_llama_server():
    global llama_process
    model_path = find_model()
    if not model_path:
        print("[WARN] No .gguf model found inside ./models directory.")
        return None

    threads, is_apple_silicon = get_system_hardware()
    ngl = "99" if is_apple_silicon else "0"

    binary = "llama-server"
    if os.path.exists("./llama-server"):
        binary = "./llama-server"

    cmd = [
        binary,
        "-m", model_path,
        "-c", "4096",
        "-t", str(max(1, threads - 1)),
        "-ngl", ngl,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "127.0.0.1"
    ]

    print(f"[INIT] Spawning inference core: {' '.join(cmd)}")
    llama_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return model_path

@app.on_event("startup")
async def startup_event():
    launch_llama_server()

@app.on_event("shutdown")
async def shutdown_event():
    global llama_process
    if llama_process:
        print("[SHUTDOWN] Terminating llama-server...")
        llama_process.terminate()

@app.get("/api/status")
async def get_status():
    threads, is_apple_silicon = get_system_hardware()
    return {
        "cpu_threads": threads,
        "gpu_detected": is_apple_silicon,
        "active_model": os.path.basename(find_model() or "None")
    }

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    body = await request.json()
    body["stream"] = True

    client = httpx.AsyncClient(timeout=120.0)

    async def event_generator():
        async with client.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body) as response:
            async for chunk in response.aiter_raw():
                yield chunk
        await client.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, log_level="info")
