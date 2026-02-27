"""
Qwen3-ASR P25 transcription server.

OpenAI-compatible API for Qwen3-ASR fine-tuned on P25 dispatch audio.
Mirrors the whisper-server API pattern for drop-in compatibility.
Supports word-level timestamps via Qwen3-ForcedAligner.

Usage:
    python server.py

    # Or with env vars:
    MODEL_PATH=/path/to/model PORT=8765 python server.py

Endpoints:
    POST /v1/audio/transcriptions  — OpenAI-compatible transcription
    GET  /v1/models                — List loaded model
    GET  /health                   — Health check
"""

import os
import tempfile
import time
from typing import Optional

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from qwen_asr import Qwen3ASRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "qwen3-asr-p25-0.6B")
ALIGNER_PATH = os.environ.get("ALIGNER_PATH", "Qwen3-ForcedAligner-0.6B")
DEVICE = os.environ.get("DEVICE", "cuda:0")
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
WORKERS = int(os.environ.get("WORKERS", "1"))

# Speech detection — reject blank/encrypted audio before wasting GPU
# RMS energy threshold: audio below this is silence/static/encrypted
# Empirically: hallucinations <0.003 RMS, real speech >0.03 RMS
SPEECH_RMS_THRESHOLD = float(os.environ.get("SPEECH_RMS_THRESHOLD", "0.01"))

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Map ISO 639-1 / common short codes to Qwen3-ASR language names
LANG_MAP = {
    "en": "English", "english": "English",
    "zh": "Chinese", "chinese": "Chinese",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "ar": "Arabic", "arabic": "Arabic",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "es": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "id": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "italian": "Italian",
    "ko": "Korean", "korean": "Korean",
    "ru": "Russian", "russian": "Russian",
    "ja": "Japanese", "japanese": "Japanese",
}

# ---------------------------------------------------------------------------
# App + model
# ---------------------------------------------------------------------------
app = FastAPI(title="qwen3-asr-p25-server", version="1.2.0")
model: Optional[Qwen3ASRModel] = None


def has_speech(audio_path: str) -> bool:
    """Check if audio contains actual speech via RMS energy.

    Rejects blank/encrypted/silent audio that would cause hallucinations.
    Empirically: blank P25 audio <0.003 RMS, real speech >0.03 RMS.
    Uses librosa to support all audio formats (wav, m4a, etc.).
    """
    audio, _ = librosa.load(audio_path, sr=None, mono=True)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms >= SPEECH_RMS_THRESHOLD


@app.on_event("startup")
def load_model():
    global model
    dt = DTYPE_MAP.get(DTYPE, torch.bfloat16)
    print(f"Loading model: {MODEL_PATH} (device={DEVICE}, dtype={DTYPE})")
    print(f"Loading aligner: {ALIGNER_PATH}")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        forced_aligner=ALIGNER_PATH,
        forced_aligner_kwargs=dict(dtype=dt, device_map=DEVICE),
        dtype=dt,
        device_map=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("Model + aligner loaded.")
    print(f"Speech detection: RMS threshold={SPEECH_RMS_THRESHOLD}")


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("qwen3-asr-p25", alias="model"),
    language: Optional[str] = Form("English"),
    response_format: str = Form("json"),
    word_timestamps: Optional[bool] = Form(None),
    timestamp_granularities: Optional[list[str]] = Form(None, alias="timestamp_granularities[]"),
):
    t0 = time.time()

    # Determine if timestamps requested
    want_timestamps = word_timestamps or bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    # Normalize language code
    lang = (language or "English").strip()
    lang = LANG_MAP.get(lang.lower(), lang)

    # Write upload to temp file
    data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Speech detection gate — skip GPU inference for blank/encrypted audio
        if not has_speech(tmp_path):
            processing_time = round(time.time() - t0, 3)
            full_text = ""
            words = []
        else:
            results = model.transcribe(
                audio=tmp_path,
                language=lang,
                return_time_stamps=want_timestamps,
            )

            r = results[0] if results else None
            full_text = r.text.strip() if r else ""
            processing_time = round(time.time() - t0, 3)

            # Build word list from timestamps
            words = []
            if want_timestamps and r and r.time_stamps:
                for item in r.time_stamps:
                    words.append({
                        "word": item.text,
                        "start": round(item.start_time, 3),
                        "end": round(item.end_time, 3),
                    })

    finally:
        os.unlink(tmp_path)

    # --- Format response ---
    if response_format == "text":
        return PlainTextResponse(full_text)

    if response_format == "verbose_json":
        resp = {
            "task": "transcribe",
            "language": lang,
            "text": full_text,
            "processing_time": processing_time,
            "model": MODEL_PATH,
        }
        if want_timestamps:
            resp["words"] = words
        return JSONResponse(resp)

    # Default: json (OpenAI-compatible)
    return JSONResponse({"text": full_text})


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "aligner": ALIGNER_PATH,
        "device": DEVICE,
        "dtype": DTYPE,
        "workers": WORKERS,
        "pid": os.getpid(),
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if WORKERS > 1:
        uvicorn.run("server:app", host=HOST, port=PORT, workers=WORKERS)
    else:
        uvicorn.run(app, host=HOST, port=PORT)
