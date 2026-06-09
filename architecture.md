# Polyglot Live — System Architecture

This document describes the high-level architecture, thread boundaries, process model, and data flows of the Polyglot Live voice companion.

---

## System Topology & Data Flow

Polyglot Live uses a multi-process, multi-threaded pipeline designed to keep E2E latency under **600ms** on desktop GPUs while supporting real-time language switching, streaming synthesis, and WebRTC-based Acoustic Echo Cancellation (AEC).

```mermaid
flowchart TB
    %% Styling Classes
    classDef hardware fill:#1f2937,stroke:#4b5563,stroke-width:2px,color:#e6edf3;
    classDef process fill:#111827,stroke:#3b82f6,stroke-width:2px,color:#3b82f6,stroke-dasharray: 5 5;
    classDef thread fill:#1f2937,stroke:#10b981,stroke-width:2px,color:#e6edf3;
    classDef external fill:#1f2937,stroke:#a855f7,stroke-width:2px,color:#e6edf3;
    classDef client fill:#1f2937,stroke:#f59e0b,stroke-width:2px,color:#e6edf3;

    %% Hardware Layer
    subgraph HW ["Physical Hardware Layer"]
        Mic["🎙️ Microphone Input<br/>(16kHz Mono)"]:::hardware
        Spk["🔊 Speaker Output<br/>(44.1kHz Stereo/Mono)"]:::hardware
        GPU["🎮 NVIDIA GPU<br/>(RTX 5050 8GB VRAM)"]:::hardware
    end

    %% Python Core Application Process
    subgraph APP ["Python Core Process (main.py)"]
        %% Mic Audio Thread
        subgraph AUD ["Audio Callback Thread"]
            MicCB["🎙️ _mic_callback()"]:::thread
            Gate["🔇 Mic Gate<br/>(If barge_in is disabled)"]:::thread
            Buffer["📥 AudioBuffer<br/>(Speaker Reference)"]:::thread
            APM["⚡ WebRTC APM Slicer<br/>(512-sample -> 10ms steps)"]:::thread
            BargeIn["⚡ Real-time Barge-in VAD<br/>(Cleaned Frame Check)"]:::thread
        end

        %% Main Pipeline Thread
        subgraph MAIN ["Main Program Thread"]
            Queue["📥 Cleaned Audio Queue<br/>(_audio_q)"]:::thread
            StateM["🔄 Pipeline State Machine<br/>(State: LISTENING, RECORDING, PROCESSING, SPEAKING)"]:::thread
            VAD["🧠 Silero VAD v5<br/>(End of Utterance Detection)"]:::thread
            Pipeline["⚙️ Pipeline Orchestrator"]:::thread
        end

        %% LLM Concurrent Thread (Streaming Mode)
        subgraph LLM_T ["LLM Streaming Thread"]
            LlmThread["🧠 llm_thread()"]:::thread
            SentQ["💬 Sentence Queue"]:::thread
        end

        %% UI Server Threads
        subgraph UI_S ["UI Server & Broadcast"]
            WebSrv["🌐 FastAPI / Uvicorn Server"]:::thread
            TermUI["📟 Rich Terminal UI Renderer"]:::thread
        end
    end

    %% External Server Processes
    subgraph SERVERS ["Independent Background Processes"]
        LlamaSrv["🦙 llama-server<br/>(Gemma-4-E2B GGUF)<br/>CUDA 13.2 GPU Runtime"]:::external
        SupertonicSrv["🎵 Supertonic V3 serve<br/>(ONNX Runtime GPU)<br/>Embedded CUDA 12 PyPI Runtime"]:::external
    end

    %% Client Layer
    subgraph CLIENT ["User Interface Layer"]
        Browser["🌐 Web Interface (WebSockets)"]:::client
    end

    %% Relations & Flows
    Mic -->|Capture Mono| MicCB
    MicCB -->|Is TTS Playing & Barge Disabled?| Gate
    Gate -->|No| APM
    
    %% Playback Reference Path
    SupertonicSrv -->|Play WAV Chunks| Spk
    SupertonicSrv -.->|on_audio_chunk callback| Buffer
    
    %% AEC processing
    Buffer -->|Read aligned reference| APM
    APM -->|process_reverse_stream + process_stream| BargeIn
    BargeIn -->|Queue clean frames| Queue
    BargeIn -->|On Barge-In: Interrupt & sets _stop_play| Pipeline
    
    Queue -->|Process frames| StateM
    StateM <-->|Evaluate Speech Prob| VAD

    %% LLM Query Flow
    Pipeline -->|Send WAV| LlmThread
    LlmThread <-->|Multimodal ASR + LLM HTTP API| LlamaSrv
    LlamaSrv -.->|Inference| GPU
    LlmThread -->|Stream words / sentences| SentQ

    %% TTS Synthesis Flow
    SentQ -->|Pop completed sentences| SupertonicSrv
    SupertonicSrv -.->|ONNX Inference| GPU

    %% UI Broadcasts
    Pipeline -->|Real-time state & Latency callbacks| UI_S
    WebSrv <-->|WebSocket| Browser
    TermUI <-->|Render updates| TermUI
```

---

## Key Architectural Modules

### 1. Multi-Threaded Audio Pipeline
To achieve full-duplex speech interaction without blocking the interface or dropping audio samples, the pipeline divides responsibilities across four distinct execution threads:
*   **Audio Callback Thread**: Runs in the context of the system's underlying audio capture server (`sounddevice` backend). It intercepts mic frames, performs WebRTC echo cancellation, runs the active barge-in check, and queues cleaned frames into `_audio_q`.
*   **Main Thread**: Loops on a tick execution cycle. It pulls frames from `_audio_q`, feeds them into Silero VAD to identify start/end of speech, and controls the state machine transition (`LISTENING` → `RECORDING` → `PROCESSING` → `SPEAKING`).
*   **LLM Streaming Thread**: When a user's speech completes, the Main Thread transitions to `PROCESSING` and fires off the LLM Query Thread. This thread consumes the token stream from `llama-server` asynchronously, splitting incoming text into individual sentences and loading them into a playout queue.
*   **TTS Output Callback / Playback**: Renders the speech chunks. If the audio callback thread flags an interruption, the TTS playout thread halts immediately (under 50ms latency).

---

### 2. WebRTC Acoustic Echo Cancellation (AEC)
Acoustic Echo Cancellation prevents the speaker output from leaking back into the microphone and triggering a self-interruption loop.
1.  **Reference Collection**: As the TTS engine plays audio blocks on the speakers, it passes them via a callback to the pipeline. The pipeline downmixes the audio to mono, resamples it from 44.1kHz to 16kHz using `torchaudio`, and appends it to a lock-protected reference `AudioBuffer`.
2.  **Chunk Boundary Matching**: WebRTC APM strictly requires **10ms frames** (160 samples at 16kHz). The system microphone records in **32ms blocks** (512 samples).
3.  **Real-Time Processing**: When a 512-sample frame is captured by the mic:
    *   We retrieve 512 samples of resampled speaker audio from the reference buffer (or zeros if nothing is playing).
    *   We slice both the mic stream and the reference stream into 160-sample slices.
    *   For each slice, we call `process_reverse_stream()` (feeding speaker audio) and `process_stream()` (feeding mic audio).
    *   We concatenate the processed slices into a cleaned output accumulator.
    *   When the accumulator reaches 512 samples, it is popped and forwarded to the VAD and barge-in validator.
4.  **Graceful Fallback**: If the compiled `aec-audio-processing` wrapper is missing, the system detects this at startup and falls back automatically to standard RMS/energy-based thresholding.

---

### 3. Latency Metrics (GPU Accelerated)

Running on an **NVIDIA RTX 5050 8GB Laptop GPU** with a mixed CUDA 12/13 runtime, the E2E latency budget is strictly tracked:

| Turn # | LLM TTFT | Total LLM Gen | TTS Latency | E2E Latency | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | 195ms | 984ms | 219ms | **867ms** | Multilingual synthesis |
| **7** | 329ms | 652ms | 167ms | **689ms** | Fast responses |
| **6** | 285ms | 1083ms | 165ms | **604ms** | Trilingual switch turn ⚡ |
| **5** | 113ms | 503ms | 167ms | **533ms** | Optimal network/GPU execution |
| **4** | 221ms | 750ms | 211ms | **794ms** | Native script Hindi response ⚡ |
| **3** | 267ms | 920ms | — | **1371ms** | Complex Hindi/English code-switch |
| **2** | 304ms | 756ms | — | **1167ms** | Fallback standard TTS playout |
| **1** | 268ms | 665ms | 153ms | **719ms** | Core English conversation turn |

*E2E latency represents the time between the user finishing speaking and the first byte of assistant speech rendering on the device.*
