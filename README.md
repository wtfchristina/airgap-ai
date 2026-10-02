# ⚡ Airgap AI: Portable Edge Inference & Offline Vector RAG

A self-contained, zero-egress neural runtime engineered for air-gapped workstations, classified environments, and local edge hardware.

Built on Apple Silicon Metal acceleration, embedded SQLite vector indexing, and zero-CDN vanilla web streaming.

---

## 🏛️ System Architecture

```text
               +--------------------------------------------------+
               |                  HOST MACHINE                    |
               |                                                  |
  Browser UI   |   +------------------------------------------+   |
[127.0.0.1:8080]<->|        FastAPI Reverse Proxy & Daemon    |   |
  (Zero CDN)   |   |                                          |   |
               |   |  - Process Supervisor (llama-server)     |   |
  VS Code      |   |  - ONNX Embedding Pipeline (FastEmbed)   |   |
[OpenAI /v1] <---->|  - Cosine Search against SQLite Vault    |   |
               |   +--------------------+---------------------+   |
               |                        |                         |
               |                        v (Unix IPC / Loopback)   |
               |   +------------------------------------------+   |
               |   |           llama.cpp Engine Core          |   |
               |   |   - Apple Metal GPU Layer Offload (-ngl) |   |
               |   |   - Multimodal Vision Projector (mmproj) |   |
               |   +--------------------+---------------------+   |
               |                        |                         |
               |                        v                         |
               |         +------------------------------+         |
               |         | Unified RAM / Metal Shaders  |         |
               |         +------------------------------+         |
               +--------------------------------------------------+
```
