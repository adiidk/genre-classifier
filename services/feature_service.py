"""Feature extraction for the web app (wraps feature_extractor.py)."""
import base64
import io
import threading

import librosa
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from feature_extractor import (
    FEATURE_COLUMNS, N_MFCC, SAMPLE_RATE, SAMPLES_PER_SEGMENT, SEGMENT_SECONDS,
    extract_features_from_signal, extract_segment_features,
)
from services import audio_service
from services.errors import AppError

MAX_SEGMENTS = 20

# (feature prefix, label, plain-English description) - shown in "Technical Details"
SCALAR_INFO = [
    ("chroma_stft", "Chroma STFT", "Energy spread over the 12 pitch classes: tonal and harmonic content."),
    ("rms", "RMS energy", "Loudness of the signal."),
    ("spectral_centroid", "Spectral centroid", "Centre of mass of the spectrum: how bright the sound is."),
    ("spectral_bandwidth", "Spectral bandwidth", "How widely the frequencies spread around the centroid."),
    ("rolloff", "Spectral rolloff", "Frequency below which 85% of the spectral energy lies."),
    ("zero_crossing_rate", "Zero-crossing rate", "How often the waveform crosses zero: noisiness and percussiveness."),
    ("harmony", "Harmonic component", "Signal level of the harmonic part (HPSS separation)."),
    ("perceptr", "Percussive component", "Signal level of the percussive part (HPSS separation)."),
]


def _summary(segments):
    df = pd.DataFrame(segments)
    scalars = []
    for base, label, description in SCALAR_INFO:
        scalars.append({
            "name": base,
            "label": label,
            "description": description,
            "mean": float(df[f"{base}_mean"].mean()),
            "var": float(df[f"{base}_var"].mean()),
        })
    return {
        "scalars": scalars,
        "tempo": float(df["tempo"].mean()),
        "mfcc_mean": [float(df[f"mfcc{i}_mean"].mean()) for i in range(1, N_MFCC + 1)],
        "mfcc_var": [float(df[f"mfcc{i}_var"].mean()) for i in range(1, N_MFCC + 1)],
    }


def _spectrogram_png(y, sr):
    """Mel spectrogram as a base64 PNG (dark theme to match the UI)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)
    db = librosa.power_to_db(mel, ref=np.max)
    duration = len(y) / sr

    bg, fg, edge = "#0b1020", "#a5b0c8", "#2a3350"
    fig = Figure(figsize=(8, 2.8), dpi=110, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
              extent=[0, duration, 0, db.shape[0]])
    ax.set_xlabel("Time (s)", color=fg)
    ax.set_ylabel("Mel band", color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(edge)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract(path):
    """Audio file -> segments (57 features each) + summary + spectrogram."""
    y = audio_service.load_audio(path)

    try:
        segments = extract_features_from_signal(y, SAMPLE_RATE)
    except Exception as exc:
        raise AppError(
            "Feature extraction failed while analysing the audio. The file may be damaged.",
            "extraction_failed", 500,
        ) from exc

    if not segments:
        raise AppError("No complete 3-second segment could be extracted.", "too_short", 422)

    for seg in segments:
        for name in FEATURE_COLUMNS:
            if not np.isfinite(seg.get(name, np.nan)):
                raise AppError(
                    "Feature extraction produced invalid values for this audio.",
                    "extraction_failed", 500,
                )

    try:
        spectrogram = _spectrogram_png(y, SAMPLE_RATE)
    except Exception:
        spectrogram = None  # the picture is optional

    return {
        "segments": segments,
        "summary": _summary(segments),
        "spectrogram": spectrogram,
        "analysis": {
            "duration_analyzed": round(len(y) / SAMPLE_RATE, 2),
            "sample_rate": SAMPLE_RATE,
            "segment_seconds": SEGMENT_SECONDS,
            "n_segments": len(segments),
            "n_features": len(FEATURE_COLUMNS),
            "samples_per_segment": SAMPLES_PER_SEGMENT,
        },
    }


def segments_to_frame(segments):
    """Validate the segments sent by the browser and build the model input."""
    if not isinstance(segments, list) or not segments:
        raise AppError("No extracted features were provided. Please analyse a file first.",
                       "bad_features", 400)
    if len(segments) > MAX_SEGMENTS:
        raise AppError("Too many segments were provided.", "bad_features", 400)

    rows = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise AppError("The feature data is malformed.", "bad_features", 400)
        row = []
        for name in FEATURE_COLUMNS:
            value = seg.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise AppError(f"Feature '{name}' is missing or invalid.", "bad_features", 400)
            row.append(float(value))
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def warm_up():
    """Run librosa once in the background so the first real request is faster."""
    def _run():
        try:
            noise = (np.random.default_rng(0).standard_normal(SAMPLES_PER_SEGMENT) * 0.1)
            extract_segment_features(noise.astype(np.float32), SAMPLE_RATE)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()