import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")                      # ADDED: lets us save plots without opening windows
import matplotlib.pyplot as plt

# Suppress standard librosa warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# 1. Define the SnS Parameters
DATASET_PATH = "genres_original"           # Folder containing the 10 genre subfolders
CSV_PATH = "music_features.csv"            # The output file for the ML part
SPEC_FOLDER = "spectrograms"               # ADDED: folder where spectrogram images are saved
SPECS_PER_GENRE = 2                        # ADDED: how many songs per genre get a saved image

SAMPLE_RATE = 22050                        # Fs = 22.05 kHz
TRACK_DURATION = 30                        # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 10                          # Slice each 30s track into 10 chunks of 3 seconds
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

SKIP_FILES = {"jazz.00054.wav"}            # ADDED: known corrupted GTZAN file


def extract_features(dataset_path):
    # This dictionary will hold all our extracted SnS features
    data = {
        "song_id": [],                     # ADDED: which song each row came from
        "length": [],
        "chroma_stft_mean": [], "chroma_stft_var": [],
        "rms_mean": [], "rms_var": [],
        "spectral_centroid_mean": [], "spectral_centroid_var": [],
        "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [],
        "rolloff_mean": [], "rolloff_var": [],
        "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [],
        "harmony_mean": [], "harmony_var": [],
        "perceptr_mean": [], "perceptr_var": [],
        "tempo": []
    }

    # Initialize the 40 MFCC columns (Mean and Variance for 20 coefficients)
    for i in range(1, 21):
        data[f"mfcc{i}_mean"] = []
        data[f"mfcc{i}_var"] = []

    data["label"] = []  # Target variable y

    os.makedirs(SPEC_FOLDER, exist_ok=True)       # ADDED
    saved_count = {}                              # ADDED: images saved so far per genre
    grid_specs = {}                               # ADDED: one spectrogram per genre for the grid figure

    # Loop through all genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        dirnames.sort()                           # ADDED: same order every run
        if dirpath != dataset_path:               # CHANGED: 'is not' -> '!='
            genre_label = os.path.basename(dirpath)   # CHANGED: works on Windows and Mac/Linux
            print(f"\nProcessing SnS computations for genre: {genre_label}")

            for f in sorted(filenames):           # CHANGED: sorted
                if not f.endswith('.wav') or f in SKIP_FILES:
                    continue
                file_path = os.path.join(dirpath, f)

                # Load the continuous signal as discrete array x[n]
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Error loading {file_path}. Skipping.")
                    continue

                # Process the 10 segments (3 seconds each)
                for d in range(NUM_SEGMENTS):
                    start = SAMPLES_PER_SEGMENT * d
                    finish = start + SAMPLES_PER_SEGMENT
                    y = signal[start:finish]  # x[n] for the current window

                    if len(y) < SAMPLES_PER_SEGMENT:   # ADDED: skip incomplete last piece
                        continue

                    # Calculate Short-Time Fourier Transform (STFT)
                    stft = np.abs(librosa.stft(y))

                    # ADDED: save spectrogram image for the report
                    # (first segment of the first few songs of each genre)
                    if d == 0 and saved_count.get(genre_label, 0) < SPECS_PER_GENRE:
                        mel = librosa.feature.melspectrogram(y=y, sr=sr)
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        fig, ax = plt.subplots(figsize=(6, 3))
                        img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                                       y_axis="mel", ax=ax)
                        fig.colorbar(img, ax=ax, format="%+2.0f dB")
                        ax.set_title(f"{genre_label} - {f}")
                        fig.tight_layout()
                        fig.savefig(os.path.join(SPEC_FOLDER, f"{f[:-4]}.png"), dpi=120)
                        plt.close(fig)
                        if genre_label not in grid_specs:
                            grid_specs[genre_label] = mel_db
                        saved_count[genre_label] = saved_count.get(genre_label, 0) + 1

                    # 0. Song id (ADDED)
                    data["song_id"].append(f)

                    # 1. Length
                    data["length"].append(len(y))

                    # 2. Chroma STFT (Pitch class energy from STFT)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    data["chroma_stft_mean"].append(np.mean(chroma_stft))
                    data["chroma_stft_var"].append(np.var(chroma_stft))

                    # 3. RMS Energy (Root Mean Square of the waveform)
                    rms = librosa.feature.rms(y=y)
                    data["rms_mean"].append(np.mean(rms))
                    data["rms_var"].append(np.var(rms))

                    # 4. Spectral Centroid (Center of mass of the Fourier spectrum)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    data["spectral_centroid_mean"].append(np.mean(spec_cent))
                    data["spectral_centroid_var"].append(np.var(spec_cent))

                    # 5. Spectral Bandwidth
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    data["spectral_bandwidth_mean"].append(np.mean(spec_bw))
                    data["spectral_bandwidth_var"].append(np.var(spec_bw))

                    # 6. Spectral Rolloff (Frequency below which 85% of energy lies)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    data["rolloff_mean"].append(np.mean(rolloff))
                    data["rolloff_var"].append(np.var(rolloff))

                    # 7. Zero Crossing Rate (Time-domain feature for noisiness)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    data["zero_crossing_rate_mean"].append(np.mean(zcr))
                    data["zero_crossing_rate_var"].append(np.var(zcr))

                    # 8. Harmonic and Percussive Source Separation
                    harmony, perceptr = librosa.effects.hpss(y)
                    data["harmony_mean"].append(np.mean(harmony))
                    data["harmony_var"].append(np.var(harmony))
                    data["perceptr_mean"].append(np.mean(perceptr))
                    data["perceptr_var"].append(np.var(perceptr))

                    # 9. Tempo (Beats per minute)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    data["tempo"].append(float(np.atleast_1d(tempo)[0]))   # CHANGED: safer

                    # 10. MFCCs (Mel-Frequency Cepstral Coefficients)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    for b in range(20):
                        data[f"mfcc{b+1}_mean"].append(np.mean(mfcc[b]))
                        data[f"mfcc{b+1}_var"].append(np.var(mfcc[b]))

                    # Append Target Label
                    data["label"].append(genre_label)

    # ADDED: one picture with a spectrogram of every genre (great for the report)
    if grid_specs:
        names = sorted(grid_specs)
        fig, axes = plt.subplots(2, 5, figsize=(20, 7))
        for ax, name in zip(axes.ravel(), names):
            librosa.display.specshow(grid_specs[name], sr=SAMPLE_RATE, x_axis="time",
                                     y_axis="mel", ax=ax)
            ax.set_title(name)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(SPEC_FOLDER, "all_genres_grid.png"), dpi=120)
        plt.close(fig)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Initializing Signals & Systems feature extraction pipeline...")
    # Generate the dataframe
    df = extract_features(DATASET_PATH)

    # Save directly to CSV for the ML part
    df.to_csv(CSV_PATH, index=False)

    print(f"\nExtraction complete! Dataset shape: {df.shape}")
    print(f"{CSV_PATH} has been created successfully.")
    print(f"Spectrogram images saved in the '{SPEC_FOLDER}' folder.")
