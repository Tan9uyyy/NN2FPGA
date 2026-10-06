# CNN MNIST

Petit réseau pour reconnaître les chiffres manuscrits, en FP32 ou quantifié.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Depuis la racine, avec l'environnement activé :

```bash
python src/train.py --bits 8
python src/evaluate.py --bits 8
python src/export_onnx.py
```

Sans `--bits`, l'entraînement et l'évaluation utilisent FP32. L'entraînement dure
5 époques. L'évaluation affiche la réussite et la taille des poids.

L'export lit les bits du fichier choisi avec `CHECKPOINT` dans `src/export_onnx.py`
et génère les deux fichiers :

```text
models/mnist_int8.onnx
       ↓ QONNX
models/mnist_int8_qonnx.onnx
```

## Fichiers

- `src/model.py` : réseau FP32.
- `src/quant_model.py` : réseau quantifié.
- `src/train.py` : entraînement et sauvegarde dans `checkpoints/`.
- `src/evaluate.py` : évaluation sur les images de test.
- `src/export_onnx.py` : export ONNX et conversion QONNX.
- `data/` : données MNIST téléchargées automatiquement.
- `models/` : modèles exportés.

La taille théorique compte les poids quantifiés et les biais FP32, hors activations
et échelles. Les checkpoints PyTorch conservent des paramètres flottants.
