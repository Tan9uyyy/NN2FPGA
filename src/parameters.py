from pathlib import Path

# Nos variables global réunis ici

def find_root(marker=".gitignore"):
    for p in Path(__file__).resolve().parents:
        if (p / marker).exists():
            return p
    raise RuntimeError(f"racine du projet introuvable ({marker})")

ROOT = find_root()
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
MODELS_DIR = ROOT / "models"

NB_EPOCH = 40 # Testé empiriquement comme amplement suffisant pour 2 à 8 bit de quantification
NB_BATCH = 64

