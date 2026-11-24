"""Quantification post-entraînement (PTQ) de YOLOv8m pré-entraîné COCO avec Brevitas.

Chaque Conv2d (BN fusionnée) devient une QuantConv2d :
  entrée  : INT<bits>, scale puissance de 2 (FixedPoint)
  poids   : INT<bits>, scale puissance de 2, par tenseur
  biais   : INT32
Les scales d'activation sont calibrées sur coco128, puis on compare FP32 vs quantifié
sur bus.jpg et en mAP sur coco128.
"""
import argparse
import json

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int32Bias
from ultralytics import YOLO

from parameters import CHECKPOINTS_DIR, DATA_DIR, ROOT

RESULTS_DIR = ROOT / "results"
COCO128 = DATA_DIR / "coco128"

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 9), default=8, help="Bits poids/activations (par défaut : 8).")
parser.add_argument("--calib", type=int, default=64, help="Nombre d'images de calibration (par défaut : 64).")
parser.add_argument("--no-val", action="store_true", help="Ne pas calculer le mAP sur coco128.")
args = parser.parse_args()


def data_yaml():
    """YAML ultralytics pointant sur data/coco128 (chemin absolu, sans toucher aux settings globaux)."""
    path = DATA_DIR / "coco128.yaml"
    names = YOLO(CHECKPOINTS_DIR / "yolov8m.pt").names
    # Le JSON est du YAML valide.
    path.write_text(json.dumps({"path": str(COCO128), "train": "images/train2017",
                                "val": "images/train2017", "names": names}))
    return str(path)


def report(yolo, tag):
    r = yolo(ROOT / "bus.jpg", verbose=False)[0]
    print(f"\n--- bus.jpg [{tag}] ---")
    for box in r.boxes:
        print(f"{r.names[int(box.cls)]:>14s}  conf={float(box.conf):.2f}  xyxy={[round(v) for v in box.xyxy[0].tolist()]}")
    RESULTS_DIR.mkdir(exist_ok=True)
    r.save(filename=str(RESULTS_DIR / f"bus_yolov8m_{tag}.jpg"))
    if args.no_val:
        return None
    m = yolo.val(data=DATA, imgsz=640, batch=8, device="cpu", plots=False, verbose=False)
    return m.box.map50, m.box.map


DATA = data_yaml()
images = sorted((COCO128 / "images" / "train2017").glob("*.jpg"))

# --- FP32 ---
fp32 = YOLO(CHECKPOINTS_DIR / "yolov8m.pt")
fp32_map = report(fp32, "fp32")

# --- Quantifié ---
yolo = YOLO(CHECKPOINTS_DIR / "yolov8m.pt")
net = yolo.model.fuse(verbose=False).eval()  # Conv+BN -> Conv (sinon BN reste en float)
bits = args.bits
conv_map = {
    torch.nn.Conv2d: (qnn.QuantConv2d, {
        "input_quant": Int8ActPerTensorFixedPoint, "input_bit_width": bits,
        "weight_quant": Int8WeightPerTensorFixedPoint, "weight_bit_width": bits,
        "bias_quant": Int32Bias, "return_quant_tensor": False}),
}
# La conv DFL (poids fixes 0..15) fait partie du post-traitement : on la laisse en float.
yolo.model = layerwise_quantize(net, compute_layer_map=conv_map, name_blacklist=["dfl"])
n_q = sum(isinstance(m, qnn.QuantConv2d) for m in yolo.model.modules())
print(f"\n{n_q} convolutions quantifiées en INT{bits}")

# Calibration en appelant le module directement : yolo.predict() travaille sur une
# deepcopy du modèle (AutoBackend), les statistiques n'arriveraient jamais ici.
letterbox = LetterBox((640, 640), auto=False)


def load(path):
    img = letterbox(image=cv2.imread(str(path)))[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    return torch.from_numpy(np.ascontiguousarray(img)).float() / 255


calib = images[:args.calib]
with torch.no_grad(), calibration_mode(yolo.model):
    for i in range(0, len(calib), 8):
        yolo.model(torch.stack([load(p) for p in calib[i:i + 8]]))
yolo.predictor = None  # au cas où : prochaine copie faite depuis le modèle calibré
print(f"Calibration sur {len(calib)} images")

q_map = report(yolo, f"int{bits}")
torch.save({"bits": bits, "state_dict": yolo.model.state_dict()}, CHECKPOINTS_DIR / f"yolov8m_int{bits}_ptq.pth")

if fp32_map:
    print(f"\n{'':8s}{'mAP50':>8s}{'mAP50-95':>10s}")
    print(f"{'FP32':8s}{fp32_map[0]:8.3f}{fp32_map[1]:10.3f}")
    print(f"{f'INT{bits}':8s}{q_map[0]:8.3f}{q_map[1]:10.3f}")
