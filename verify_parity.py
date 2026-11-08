"""
Check that feature_extractor.py reproduces the values stored in your CSV.

    python verify_parity.py path/to/blues.00000.wav
    python verify_parity.py path/to/blues.00000.wav --song-id 0

By default it compares with the first song in the CSV.
"""
import argparse

import librosa
import numpy as np
import pandas as pd

import config
from feature_extractor import (
    FEATURE_COLUMNS, SAMPLE_RATE, TRACK_DURATION, extract_features_from_signal,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to a .wav file that is in your dataset")
    parser.add_argument("--csv", default=str(config.DEFAULT_CSV))
    parser.add_argument("--song-id", default=None, help="song_id of this file in the CSV")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True, duration=TRACK_DURATION)
    segments = extract_features_from_signal(y, sr)
    mine = pd.DataFrame(segments)[FEATURE_COLUMNS]

    df = pd.read_csv(args.csv)
    song_id = args.song_id if args.song_id is not None else df["song_id"].iloc[0]
    rows = df[df["song_id"].astype(str) == str(song_id)]
    if rows.empty:
        raise SystemExit(f"song_id {song_id!r} not found in the CSV.")

    theirs = rows[FEATURE_COLUMNS].head(len(mine)).reset_index(drop=True)
    n = min(len(mine), len(theirs))
    mine = mine.head(n)
    theirs = theirs.head(n)

    rel = (mine.values - theirs.values) / (np.abs(theirs.values) + 1e-9)
    worst_per_col = pd.Series(np.abs(rel).max(axis=0), index=FEATURE_COLUMNS)

    print(f"Compared {n} segments of song_id={song_id} x {len(FEATURE_COLUMNS)} features\n")
    print("Largest relative differences:")
    print(worst_per_col.sort_values(ascending=False).head(8).round(6).to_string())

    check = worst_per_col.drop(labels=["tempo"])
    print()
    if (check < 1e-2).all():
        print("PASS: features match your CSV (tempo may differ slightly, that is normal).")
    else:
        print("MISMATCH: some features differ. Send me this output together with your")
        print("legacy/extractfeatures.py and I will align feature_extractor.py exactly.")


if __name__ == "__main__":
    main()