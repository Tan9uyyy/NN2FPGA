import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from quant_model import QuantCNN

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 33), help="Sans option : FP32.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
precision = "fp32" if args.bits is None else f"int{args.bits}"
checkpoint = ROOT / "checkpoints" / f"mnist_{precision}.pth"
if not checkpoint.is_file():
    parser.error(f"Poids absents : entraîne d'abord le modèle {precision.upper()}.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN() if args.bits is None else QuantCNN(args.bits)
model = model.to(device)
saved = torch.load(checkpoint, map_location=device, weights_only=True)
if "state_dict" in saved and saved["bits"] != args.bits:
    parser.error("Le nombre de bits ne correspond pas aux poids sauvegardés.")
model.load_state_dict(saved.get("state_dict", saved))
model.eval()

dataset = datasets.MNIST(
    root=str(ROOT / "data"), train=False, download=True, transform=transforms.ToTensor(),
)
loader = DataLoader(dataset, batch_size=64)
correct = 0
with torch.no_grad():
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += (predictions == labels).sum().item()

# Estimation : poids compactés au nombre de bits choisi, biais en FP32.
# Elle exclut les activations et les échelles de quantification.
weights = sum(p.numel() for name, p in model.named_parameters()
              if name.endswith("weight"))
biases = sum(p.numel() for name, p in model.named_parameters()
             if name.endswith("bias"))
size = weights * (args.bits or 32) / 8 + biases * 4
print(f"Modèle : {precision.upper()}")
print(f"Réussite : {100 * correct / len(dataset):.2f} % ({correct}/{len(dataset)})")
print(f"Poids + biais théoriques : {size / 1024:.2f} Kio")
print(f"Fichier .pth réel : {checkpoint.stat().st_size / 1024:.2f} Kio")
