# 🌍 Polyglot Live — Real-Time Multilingual Voice Companion

> **Polyglot Live** is an ultra-low-latency, real-time voice agent designed for seamless multilingual support. Built with standard-setting WebRTC Acoustic Echo Cancellation (AEC), it detects language switches mid-sentence in under one utterance and responds dynamically in English, Hindi, and Spanish without losing conversation context.

---

## 🚀 Key Features

*   **⚡ Real-Time Multilingual Speech-to-Speech (S2S)**: Switches languages mid-dialogue. Supports English (en), Hindi (hi), and Spanish (es).
*   **🎙️ WebRTC Acoustic Echo Cancellation (AEC)**: Filters out speaker reference loopback using WebRTC's Audio Processing Module (APM), allowing clean barge-in even in echo-prone environments.
*   **🔇 Intelligent Mic Gating**: Disables the microphone input during TTS playback when barge-in is disabled, preventing self-interruption and deferred audio loops.
*   **🎬 Simulated Demo Mode**: Automated turn-by-turn simulation using evaluation scripts.
    *   **Native Transliteration**: Automatically transliterates Hinglish/Romanized text to Devanagari script using strict LLM-reasoning (thinking mode) for natural-sounding TTS.
    *   **Automated Playback**: Injects synthesized user speech directly into the audio pipeline, generating user/agent WAV files for analysis under `logs/demo_audio/`.
*   **🚄 Streaming TTS Engine**: Drains token blocks sentence-by-sentence, allowing the first sentence to play while the LLM generates subsequent text, slashing perceived latency.
*   **🏎️ GPU-Accelerated Pipelines**: Configured for CUDA GPU acceleration for both the LLM and the Supertonic TTS model.

---

## 🛠️ Technology Stack

| Stage | Component | Description / Advantage |
| :--- | :--- | :--- |
| **VAD** | **Silero VAD v5** | Run locally on 512-sample frames (32ms blocks). High-speed voice activity detection. |
| **ASR + LLM** | **Gemma 4 E2B Q8** | Runs via `llama-server` on GPU. Single unified model handles speech transcription and contextual response. |
| **TTS** | **Supertonic V3** | Local ONNX Runtime engine. Synthesizes high-fidelity voice output in 31 languages (including native Hindi). |
| **AEC** | **WebRTC APM** | Real-time echo cancellation via `aec-audio-processing` wrapper. |

---

## 📊 Latency Benchmarks (RTX 5050 8GB VRAM)

Below are actual E2E performance stats captured on an **NVIDIA RTX 5050 8GB Laptop GPU** running ONNX Runtime GPU (CUDA 12/13 mix):

| Turn | LLM TTFT | Total LLM Gen | TTS Latency | Total E2E | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 268ms | 665ms | 153ms | **719ms** | English turn, streaming TTS |
| **2** | 304ms | 756ms | — | **1167ms** | Fallback standard TTS invocation |
| **3** | 267ms | 920ms | — | **1371ms** | English/Hindi code-switch (thinking) |
| **4** | 221ms | 750ms | 211ms | **794ms** | Hindi response (Devanagari) ⚡ |
| **5** | 113ms | 503ms | 167ms | **533ms** | Highly optimized response |
| **6** | 285ms | 1083ms | 165ms | **604ms** | Tri-lingual stress turn ⚡ |
| **7** | 329ms | 652ms | 167ms | **689ms** | Fast turn transitions |
| **8** | 195ms | 984ms | 219ms | **867ms** | Clean multilingual close |

*Note: TTFT = Time to First Token. E2E indicates elapsed time between end of user speech and the start of agent audio playback.*

---

## ⚡ Architecture Deep Dive

### 1. WebRTC Echo Cancellation (AEC) & Audio Resampling
Real-time echo cancellation requires comparing the recorded microphone input with the reference speaker playback (what the user hears).
*   **Sample Rate Matching**: Supertonic outputs audio at 44.1kHz. Silero VAD and WebRTC APM process mic input at 16kHz. We resample the playout stream to 16kHz mono using `torchaudio` and feed it into a lock-protected, thread-safe `AudioBuffer`.
*   **10ms Chunk Alignment**: WebRTC APM requires strictly **10ms frames** (160 samples at 16kHz). Our microphone frame size is **32ms** (512 samples). We implement a sample accumulator that slices incoming 512-sample microphone frames into 10ms steps, feeds them alongside aligned speaker reference chunks to WebRTC AEC (`process_reverse_stream` + `process_stream`), and re-assembles the cleaned output into a 512-sample frame for VAD evaluation.

```
       Playout Reference (44.1kHz float32)
                       │
                       ▼
       ┌───────────────────────────────┐
       │   torchaudio Resampler        │  ──► Resamples to 16kHz mono int16
       └───────────────────────────────┘
                       │
                       ▼
         Thread-safe Playback Buffer
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             Microphone Callback             │
│                                             │
│  [ Mic Frame (512) ]    [ Ref Frame (512) ] │
│           │                      │          │
│           └──────────┬───────────┘          │
│                      ▼                      │
│        Accumulate in 160-sample slices      │
│                      │                      │
│                      ▼                      │
│        WebRTC Audio Processing (APM)        │
│          ├─ process_reverse_stream(ref)     │
│          └─ process_stream(mic)             │
│                      │                      │
│                      ▼                      │
│        Reconstruct 512-sample Clean Frame   │
└─────────────────────────────────────────────┘
                       │
                       ▼
            Downstream VAD Queue
```

### 2. Multi-Threaded Flow & Interruption
*   **Audio Capture Thread**: Runs in the `sounddevice` InputStream callback. Discards microphone input entirely when `barge_in` is disabled and TTS is active. If barge-in is enabled, it computes VAD probability on the AEC-cleaned frame.
*   **Main Thread**: Manages the pipeline state machine (`LISTENING` ──► `RECORDING` ──► `PROCESSING` ──► `SPEAKING`).
*   **LLM Streaming Thread**: Asynchronously queries the `llama-server`. Aggregates tokens, groups them into completed sentences, and pushes them to a `Sentence Queue`.
*   **TTS Playout Thread**: Pulls sentences from the queue and synthesizes them. If the capture thread registers a barge-in (VAD probability threshold exceeded), it sets `_stop_play`, causing the TTS player loop to abort playback within 50ms.

---

## ⚙️ Quick Start

### 1. Prerequisites
Install `swig`, `meson`, and C/C++ build tools required to compile `aec-audio-processing` from source (if no wheel is available for your platform):
```bash
sudo apt-get install swig build-essential meson
```

### 2. Installation
```bash
# Clone the repository and navigate inside
cd polyglot

# Activate your virtual environment and install requirements
pip install -r requirements.txt
```

### 3. Start the Inference Servers
Review paths in `config.yaml` to ensure they point to your local GGUF models, then start the LLM/TTS backends:
```bash
chmod +x start_servers.sh
./start_servers.sh start
```

### 4. Running the Companion
To launch the terminal dashboard:
```bash
python main.py
```

To launch the web interface (FastAPI + WebSockets at `http://localhost:7860`):
```bash
python main.py --web
```

To stop the background server processes:
```bash
./start_servers.sh stop
```

---

## 🎬 Demo Mode (Evaluation Helper)
Since humans cannot speak three languages natively in quick succession during validation, **Demo Mode** acts as a simulator:
1.  **Strict Transliteration**: Translates Hinglish scripts (e.g. `"Theek hai, mujhe aur..."`) to Devanagari Hindi using Gemma 4's reasoning/thinking steps to achieve natural-sounding TTS.
2.  **Audio Injection**: Synthesizes the user's turn, plays it over the speakers, and injects the PCM samples directly into the input queue to trigger natural VAD and pipeline flows.
3.  **Logs**: User and Assistant WAV audio for every turn are logged sequentially in `logs/demo_audio/`.
