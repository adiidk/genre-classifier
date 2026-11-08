"""
Feature extraction for the web app.

Mirrors the recipe of the original extractfeatures.py:
  * audio resampled to 22,050 Hz, mono
  * first 30 seconds, cut into 3-second segments (66,150 samples each)
  * 57 features per segment
"""
import numpy as np
import librosa

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_SECONDS   # 66150
TRACK_DURATION = 30
NUM_SEGMENTS = TRACK_DURATION // SEGMENT_SECONDS      # 10
N_MFCC = 20

_BASE_FEATURES = [
    "chroma_stft",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate",
    "harmony",
    "perceptr",
]


def _build_columns():
    cols = []
    for name in _BASE_FEATURES:
        cols.append(f"{name}_mean")
        cols.append(f"{name}_var")
    cols.append("tempo")
    for i in range(1, N_MFCC + 1):
        cols.append(f"mfcc{i}_mean")
        cols.append(f"mfcc{i}_var")
    return cols


# 8*2 + 1 + 20*2 = 57 columns, in a fixed order
FEATURE_COLUMNS = _build_columns()


def extract_segment_features(y, sr=SAMPLE_RATE):
    """Return a dict with the 57 features of ONE audio segment."""
    f = {}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["chroma_stft_mean"] = float(np.mean(chroma))
    f["chroma_stft_var"] = float(np.var(chroma))

    rms = librosa.feature.rms(y=y)
    f["rms_mean"] = float(np.mean(rms))
    f["rms_var"] = float(np.var(rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    f["spectral_centroid_mean"] = float(np.mean(centroid))
    f["spectral_centroid_var"] = float(np.var(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    f["spectral_bandwidth_var"] = float(np.var(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    f["rolloff_mean"] = float(np.mean(rolloff))
    f["rolloff_var"] = float(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    f["zero_crossing_rate_mean"] = float(np.mean(zcr))
    f["zero_crossing_rate_var"] = float(np.var(zcr))

    y_harm, y_perc = librosa.effects.hpss(y)
    f["harmony_mean"] = float(np.mean(y_harm))
    f["harmony_var"] = float(np.var(y_harm))
    f["perceptr_mean"] = float(np.mean(y_perc))
    f["perceptr_var"] = float(np.var(y_perc))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f["tempo"] = float(np.atleast_1d(tempo)[0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        f[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        f[f"mfcc{i + 1}_var"] = float(np.var(mfcc[i]))

    return f


def extract_features_from_signal(y, sr=SAMPLE_RATE, num_segments=NUM_SEGMENTS):
    """Cut a signal into 3 s segments and return a list of feature dicts."""
    segments = []
    for i in range(num_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        chunk = y[start:end]
        if len(chunk) < SAMPLES_PER_SEGMENT:
            break  # ignore an incomplete last piece
        segments.append(extract_segment_features(chunk, sr))
    return segments