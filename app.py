import os
import sys
import glob
import subprocess
import multiprocessing
import platform
import time
import httpx
from pypdf import PdfReader
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Airgap AI")

LLAMA_SERVER_PORT = 8081
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
llama_process = None
current_model_file = None

def get_system_hardware():
    threads = multiprocessing.cpu_count()
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return threads, is_apple_silicon

def get_available_models():
    files = glob.glob("models/*.gguf")
    return [os.path.basename(f) for f in files]

def stop_llama_server():
    global llama_process
    if llama_process:
        print("[PROCESS] Stopping current llama-server instance...")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=5)
        except Exception:
            llama_process.kill()
        llama_process = None

def start_llama_server(model_filename: str):
    global llama_process, current_model_file
    model_path = os.path.join("models", model_filename)
    if not os.path.exists(model_path):
        return False

    stop_llama_server()

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

    print(f"[LAUNCH] Loading {model_filename} on Metal GPU...")
    llama_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_model_file = model_filename
    time.sleep(1.5)
    return True

@app.on_event("startup")
async def startup_event():
    models = get_available_models()
    if models:
        # Default to 3b if present, else first available
        target = "llama-3.2-3b.gguf" if "llama-3.2-3b.gguf" in models else models[0]
        start_llama_server(target)

@app.on_event("shutdown")
async def shutdown_event():
    stop_llama_server()

@app.get("/api/status")
async def get_status():
    threads, is_apple_silicon = get_system_hardware()
    return {
        "cpu_threads": threads,
        "gpu_detected": is_apple_silicon,
        "active_model": current_model_file or "None",
        "available_models": get_available_models()
    }

@app.post("/api/switch-model")
async def switch_model(payload: dict):
    model_name = payload.get("model")
    if not model_name or model_name not in get_available_models():
        return JSONResponse({"status": "error", "message": "Model file not found"}, status_code=400)
    success = start_llama_server(model_name)
    return {"status": "success", "active_model": current_model_file}

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Reads PDF directly from RAM buffer with zero external network access"""
    try:
        reader = PdfReader(file.file)
        text_content = ""
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            text_content += f"\n--- Page {i+1} ---\n" + extracted

        # Truncate context to ~12,000 characters to prevent context-window overflow
        truncated = text_content[:12000]
        return {
            "status": "success",
            "filename": file.filename,
            "text": truncated,
            "pages": len(reader.pages)
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    body = await request.json()
    body["stream"] = True

    client = httpx.AsyncClient(timeout=180.0)

    async def event_generator():
        try:
            async with client.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body) as response:
                async for chunk in response.aiter_raw():
                    yield chunk
        finally:
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
