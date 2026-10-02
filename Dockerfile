# Stage 1: Build llama.cpp binary from source
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git . && \
    cmake -B build -DGGML_AVX2=ON -DGGML_AVX512=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-server

# Stage 2: Runtime image
FROM python:3.11-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache offline embedding model inside container image
RUN python3 -c \"from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')\"

COPY app.py .
COPY static/ static/

EXPOSE 8080

CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]
