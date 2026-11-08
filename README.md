# Music Genre Classifier

Signals & Systems project: audio feature extraction (librosa) + ML genre prediction
(Logistic Regression, SVM) with a Flask web interface.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run
```
python app.py
```
Open http://127.0.0.1:5000

## Retrain models (optional)
```
python train_models.py            # full, with cross-validation
python train_models.py --skip-cv  # quick
```
Training uses `data/music_features.csv`, generated from the GTZAN dataset
(not included in this repo because of its size).

## Notes
- Upload .wav or .mp3. If MP3 files fail to load, install ffmpeg.
- Only a 30-second portion of each upload is analysed.