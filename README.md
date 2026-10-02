# ⚡ Airgap AI

A self-contained, portable, zero-egress edge AI runtime engineered for air-gapped systems and privacy-critical environments.

Runs local quantized LLMs directly on hardware with zero external network calls, zero CDN dependencies, and native streaming via Server-Sent Events (SSE).

---

## 🚀 Key Features

* **True Air-Gap Compliance:** Runs completely offline without external telemetry or outbound network requests.
* **Apple Silicon & CPU Hardware Auto-Negotiation:** Offloads inference to Metal GPU on Mac M-series chips or optimizes CPU threads automatically.
* **Zero-CDN Native UI:** Embedded, vanilla HTML/CSS/JS frontend with native browser `ReadableStream` decoding for token-by-token generation.
* **OpenAI-Compatible Gateway:** Proxies requests via an internal FastAPI daemon supporting standard chat completion payloads.
* **Quantized Edge Efficiency:** Built to run compact GGUF weights (1B–7B models) with minimal memory overhead.

---

## 🛠️ Quickstart

### 1. Prerequisites
Ensure `llama.cpp` is installed:
\`\`\`bash
brew install llama.cpp
\`\`\`

### 2. Download Model Weights
Place any `.gguf` file inside the `models/` directory:
\`\`\`bash
curl -L -o models/llama-3.2-1b.gguf "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
\`\`\`

### 3. Launch
Run the automated launcher:
\`\`\`bash
./run.sh
\`\`\`

Open \`http://127.0.0.1:8080\` in your browser.

---

## 🔒 Security & Verification

To verify full offline capability:
1. Disconnect all network interfaces (disable Wi-Fi / Ethernet).
2. Submit prompts via the local web console at \`127.0.0.1:8080\`.
3. All token generation occurs strictly in local system RAM/VRAM.
