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
python src/export_onnx.py --bits 8
python src/onnx_inference_cnn.py --bits 8 --samples 100
```

`--bits` accepte de 2 à 8 : l'export ONNX standard (QCDQ) ne gère pas plus de 8 bits.
Sans `--bits`, l'entraînement et l'évaluation utilisent FP32. L'entraînement dure
5 époques. L'évaluation affiche la réussite et la taille des poids.

L'export utilise l'option `--bits` (par défaut 8) et génère les deux fichiers :

```text
models/mnist_int8.onnx
       ↓ QONNX
models/mnist_int8_qonnx.onnx
```

`onnx_inference_cnn.py` charge le premier fichier avec ONNX Runtime sur CPU.
Le script exécute le modèle sur des images de test, compare les prédictions aux
étiquettes et affiche le temps total, le temps moyen par image et le débit.
Comme dans l'exemple CIFAR-10, les optimisations du graphe ONNX Runtime sont
désactivées. Sans `--samples`, le script parcourt les 10 000 images de test
MNIST. Sans `--bits`, il utilise le modèle INT8.
[Exemple CIFAR-10 de nn2FPGA](https://github.com/robertoBosio/nn2FPGA/blob/main/config_examples/cifar10/onnx_inference_cifar10.py)

## Fichiers

- `src/model.py` : réseau FP32.
- `src/quant_model.py` : réseau quantifié.
- `src/train.py` : entraînement et sauvegarde dans `checkpoints/`.
- `src/evaluate.py` : évaluation sur les images de test.
- `src/export_onnx.py` : export ONNX et conversion QONNX.
- `src/onnx_inference_cnn.py` évalue le modèle ONNX exporté sur MNIST.
- `data/` : données MNIST téléchargées automatiquement.
- `models/` : modèles exportés.

La taille théorique compte les poids quantifiés et les biais FP32, hors activations
et échelles. Les checkpoints PyTorch conservent des paramètres flottants.
