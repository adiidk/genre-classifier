"""Central configuration for the Music Genre Classification app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "music_features.csv"

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_UPLOAD_MB = 50

RANDOM_STATE = 42

HOST = "127.0.0.1"
PORT = 5000