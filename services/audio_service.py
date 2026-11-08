"""Temporary upload handling and audio loading / validation."""
import contextlib
import uuid
from pathlib import Path

import librosa
import numpy as np

import config
from feature_extractor import SAMPLE_RATE, SEGMENT_SECONDS, TRACK_DURATION
from services.errors import AppError


def check_extension(filename):
    if not filename:
        raise AppError("No file was selected.", "no_file", 400)
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        shown = ext if ext else "unknown"
        raise AppError(
            f"Unsupported file type ({shown}). Please upload a .wav or .mp3 file.",
            "unsupported_type", 415,
        )
    return ext


@contextlib.contextmanager
def temporary_upload(file_storage):
    """Save an upload under a random name, and ALWAYS delete it afterwards."""
    if file_storage is None or not file_storage.filename:
        raise AppError("No file was uploaded.", "no_file", 400)

    ext = check_extension(file_storage.filename)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        file_storage.save(path)
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def cleanup_uploads():
    """Remove leftovers from a previous crash (called once at startup)."""
    if not config.UPLOAD_DIR.exists():
        return
    for item in config.UPLOAD_DIR.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            try:
                item.unlink()
            except OSError:
                pass


def inspect_audio(path):
    """Quick look at a file: sample rate and duration (no full decode)."""
    try:
        sample_rate = int(librosa.get_samplerate(str(path)))
        duration = float(librosa.get_duration(path=str(path)))
    except Exception as exc:
        raise AppError(
            "This audio file could not be read. It may be corrupt or use an unsupported encoding.",
            "corrupt_audio", 422,
        ) from exc

    if duration < SEGMENT_SECONDS:
        raise AppError(
            f"The audio is too short ({duration:.1f} s). At least {SEGMENT_SECONDS} seconds are needed.",
            "too_short", 422,
        )

    return {
        "sample_rate": sample_rate,
        "duration": round(duration, 2),
        "will_be_truncated": duration > TRACK_DURATION + 0.5,
        "analysis_seconds": round(min(duration, TRACK_DURATION), 2),
    }


def load_audio(path):
    """Load mono audio at 22,050 Hz, first 30 seconds only."""
    try:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=TRACK_DURATION)
    except Exception as exc:
        raise AppError(
            "This audio file could not be decoded. It may be corrupt or use an unsupported encoding.",
            "corrupt_audio", 422,
        ) from exc

    if y.size < SEGMENT_SECONDS * SAMPLE_RATE:
        raise AppError(
            f"The audio is too short. At least {SEGMENT_SECONDS} seconds are needed.",
            "too_short", 422,
        )
    if not np.all(np.isfinite(y)):
        raise AppError("The audio contains invalid samples (corrupt file).", "corrupt_audio", 422)
    if float(np.max(np.abs(y))) < 1e-4:
        raise AppError("The audio is silent, so there is nothing to analyse.", "silent_audio", 422)
    return y