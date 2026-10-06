import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from quant_model import QuantCNN

# Sans --bits : FP32. Avec --bits 8 : modèle quantifié sur 8 bits.
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 33), help="Sans option : FP32.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
precision = "fp32" if args.bits is None else f"int{args.bits}"
checkpoint = ROOT / "checkpoints" / f"mnist_{precision}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN() if args.bits is None else QuantCNN(args.bits)
model = model.to(device)
print(f"Modèle : {precision.upper()} | Device : {device}")

dataset = datasets.MNIST(
    root=str(ROOT / "data"), train=True, download=True, transform=transforms.ToTensor(),
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    print(f"Epoch {epoch + 1}/{epochs} - Loss : {total_loss / len(dataset):.4f}")

checkpoint.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "bits": args.bits, "epochs": epochs}, checkpoint)
print(f"Poids sauvegardés : {checkpoint}")
