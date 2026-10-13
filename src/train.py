import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from paths import CHECKPOINTS_DIR, DATA_DIR
from quant_model import QuantCNN

# Sans --bits : FP32. Avec --bits 8 : modèle quantifié sur 8 bits.
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 9), help="Sans option : FP32.")
args = parser.parse_args()

precision = "fp32" if args.bits is None else f"int{args.bits}"
checkpoint = CHECKPOINTS_DIR / f"mnist_{precision}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN() if args.bits is None else QuantCNN(args.bits)
model = model.to(device)
print(f"Modèle : {precision.upper()} | Device : {device}")

dataset = datasets.MNIST(
    root=str(DATA_DIR), train=True, download=True, transform=transforms.ToTensor(),
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Nepochs = 10
losses = []

model.train()
for epoch in range(Nepochs):
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    epoch_loss = total_loss / len(dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{Nepochs} - Loss : {epoch_loss:.4f}")

checkpoint.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "bits": args.bits, "epochs": Nepochs}, checkpoint)
print(f"Poids sauvegardés : {checkpoint}")

plt.plot(range(1, Nepochs + 1), losses, marker="o")
plt.xlabel("Époque")
plt.ylabel("Loss moyenne")
plt.title(f"Entraînement MNIST — {precision.upper()}")
plt.grid(True)
graph = checkpoint.parent / f"mnist_{precision}_loss.png"
plt.savefig(graph)
print(f"Courbe sauvegardée : {graph}")
plt.show()
