import os
import sys
import glob
import json
import sqlite3
import subprocess
import multiprocessing
import platform
import time
import numpy as np
import httpx
from pypdf import PdfReader
from fastembed import TextEmbedding
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Airgap AI")

LLAMA_SERVER_PORT = 8081
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
DB_PATH = "airgap_vault.db"
llama_process = None
current_model_file = None

print("[INIT] Loading local embedding weights (bge-small-en-v1.5)...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_system_hardware():
    threads = multiprocessing.cpu_count()
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return threads, is_apple_silicon

def get_available_models():
    # Only return primary GGUFs, exclude standalone projector files
    files = glob.glob("models/*.gguf")
    return [os.path.basename(f) for f in files if "mmproj" not in f]

def stop_llama_server():
    global llama_process
    if llama_process:
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

    # Multimodal projector auto-link
    base_prefix = model_filename.split("-")[0]
    potential_projectors = glob.glob(f"models/{base_prefix}*mmproj*.gguf")
    if potential_projectors:
        print(f"[VISION] Attaching multimodal projector: {potential_projectors[0]}")
        cmd.extend(["--mmproj", potential_projectors[0]])

    print(f"[LAUNCH] Executing inference daemon: {' '.join(cmd)}")
    llama_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_model_file = model_filename
    time.sleep(2)
    return True

@app.on_event("startup")
async def startup_event():
    models = get_available_models()
    if models:
        target = "llama-3.2-3b.gguf" if "llama-3.2-3b.gguf" in models else models[0]
        start_llama_server(target)

@app.on_event("shutdown")
async def shutdown_event():
    stop_llama_server()

@app.get("/api/status")
async def get_status():
    threads, is_apple_silicon = get_system_hardware()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks")
    chunk_count = cur.fetchone()[0]
    conn.close()

    return {
        "cpu_threads": threads,
        "gpu_detected": is_apple_silicon,
        "active_model": current_model_file or "None",
        "available_models": get_available_models(),
        "indexed_chunks": chunk_count
    }

@app.post("/api/switch-model")
async def switch_model(payload: dict):
    model_name = payload.get("model")
    if not model_name or model_name not in get_available_models():
        return JSONResponse({"status": "error", "message": "Model not found"}, status_code=400)
    start_llama_server(model_name)
    return {"status": "success", "active_model": current_model_file}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks

@app.post("/api/index-document")
async def index_document(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        chunks = chunk_text(full_text)

        if not chunks:
            return JSONResponse({"status": "error", "message": "No extractable text found"}, status_code=400)

        embeddings = list(embed_model.embed(chunks))

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO document_chunks (filename, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
                (file.filename, i, chunk, json.dumps(emb.tolist()))
            )
        conn.commit()
        conn.close()

        return {"status": "success", "filename": file.filename, "chunks_indexed": len(chunks)}
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"{type(e).__name__}: {str(e)}"}, status_code=500)

@app.post("/api/clear-vault")
async def clear_vault():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

def retrieve_relevant_context(query: str, top_k: int = 3):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filename, content, embedding FROM document_chunks")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return ""

    query_emb = list(embed_model.embed([query]))[0]

    scored = []
    for filename, content, emb_json in rows:
        doc_emb = np.array(json.loads(emb_json))
        score = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored.append((score, filename, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:top_k]

    return "\n\n".join([f"[{fn}]: {text}" for _, fn, text in top_chunks])

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Text RAG enrichment if last message is pure text
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
        user_query = messages[-1]["content"]
        relevant_context = retrieve_relevant_context(user_query)
        if relevant_context:
            messages[-1]["content"] = (
                f"Context from indexed offline documents:\n\"\"\"\n{relevant_context}\n\"\"\"\n\n"
                f"User Question: {user_query}\nAnswer strictly using the provided context if possible."
            )

    body["messages"] = messages
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
