import argparse
import time

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from parameters import DATA_DIR, MODELS_DIR


def mnist_dataloader(sample_size: int | None):
    test_data = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    if sample_size is not None:
        test_data = Subset(test_data, range(min(sample_size, len(test_data))))

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)


def postprocess(outputs, labels: torch.Tensor) -> int:
    logits = np.asarray(outputs[0])
    if logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)

    predicted = np.argmax(logits, axis=1).reshape(-1)
    labels_np = labels.cpu().numpy().reshape(-1)
    return int(np.sum(predicted == labels_np))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bits", type=int, choices=range(2, 9), default=8,
        help="Nombre de bits du modèle exporté (défaut : 8).",
    )
    parser.add_argument(
        "--samples", type=int,
        help="Nombre d'images de test (par défaut : tout MNIST).",
    )
    args = parser.parse_args()
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples doit être strictement positif.")

    model_path = MODELS_DIR / f"mnist_int{args.bits}.onnx"
    if not model_path.is_file():
        parser.error(f"Modèle absent : exporte d'abord le modèle INT{args.bits}.")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    test_loader = mnist_dataloader(args.samples)

    total_correct = 0
    total_samples = 0
    total_time = 0.0

    for features, labels in test_loader:
        input_data = {input_name: features.numpy().astype(np.float32)}

        start = time.perf_counter()
        outputs = session.run(None, input_data)
        total_time += time.perf_counter() - start

        total_correct += postprocess(outputs, labels)
        total_samples += labels.shape[0]

    accuracy = total_correct / total_samples if total_samples else 0.0
    average_image_time = total_time / total_samples if total_samples else 0.0
    throughput = total_samples / total_time if total_time > 0 else 0.0

    print("\n===== Résultats finaux =====")
    print(f"Échantillons traités : {total_samples}")
    print(f"Prédictions correctes : {total_correct}")
    print(f"Précision :            {accuracy:.4f}")
    print(f"Temps total :          {total_time:.6f} s")
    print(f"Temps moyen/image :    {average_image_time:.6f} s")
    print(f"Débit :                {throughput:.2f} images/s")


if __name__ == "__main__":
    main()
