"""Export ONNX de YOLOv8m, une fois pour toutes, dans models/.

  python src/export_yolov8m_onnx.py               # FP32 + INT8
  python src/export_yolov8m_onnx.py --bits 4      # FP32 + INT4 (lancer d'abord yolov8m_quant.py --bits 4)
  python src/export_yolov8m_onnx.py --no-fp32

Fichiers produits (entrée [1, 3, 640, 640] float 0-1, sortie [1, 84, 8400]) :
  models/yolov8m_fp32.onnx           référence flottante (ONNX Runtime CPU)
  models/yolov8m_int<b>.onnx         quantifié, format QCDQ (QuantizeLinear/DequantizeLinear)
  models/yolov8m_int<b>_qonnx.onnx   quantifié, opérateurs Quant QONNX -> entrée du compilateur nn2FPGA
"""
import argparse
import shutil

import torch
from brevitas.export import export_onnx_qcdq
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.util.cleanup import cleanup_model
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from parameters import MODELS_DIR
from yolov8m_brevitas import FP32_WEIGHTS, build_quant_yolo, ptq_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 9), default=8, help="Bits du modèle quantifié (défaut : 8).")
parser.add_argument("--no-fp32", action="store_true", help="Ne pas exporter le modèle FP32.")
args = parser.parse_args()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- FP32 : exporteur ultralytics (écrit à côté du .pt, on déplace dans models/) ---
if not args.no_fp32:
    exported = YOLO(FP32_WEIGHTS).export(format="onnx", imgsz=640, opset=13, simplify=False, dynamic=False)
    fp32_out = MODELS_DIR / "yolov8m_fp32.onnx"
    shutil.move(exported, fp32_out)
    print("Export FP32 terminé :", fp32_out)

# --- Quantifié : Brevitas QCDQ puis QONNX ---
checkpoint = ptq_checkpoint(args.bits)
if not checkpoint.is_file():
    parser.error(f"Poids absents : lance d'abord  python src/yolov8m_quant.py --bits {args.bits}")
saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
if saved.get("bits") != args.bits:
    parser.error("Le nombre de bits ne correspond pas aux poids sauvegardés.")

model = build_quant_yolo(args.bits).model
model.load_state_dict(saved["state_dict"])
model.eval()
# Tête Detect en mode export : une seule sortie [1, 84, 8400] (boîtes décodées + scores sigmoïde).
for m in model.modules():
    if isinstance(m, Detect):
        m.export, m.format, m.dynamic = True, "onnx", False

qcdq_out = MODELS_DIR / f"yolov8m_int{args.bits}.onnx"
qonnx_out = MODELS_DIR / f"yolov8m_int{args.bits}_qonnx.onnx"
with torch.no_grad():
    export_onnx_qcdq(model, torch.zeros(1, 3, 640, 640), export_path=str(qcdq_out), dynamo=False,
                     keep_initializers_as_inputs=False, opset_version=13,
                     input_names=["images"], output_names=["output0"])
print("Export QCDQ terminé :", qcdq_out)

qonnx_model = cleanup_model(ModelWrapper(str(qcdq_out)))
qonnx_model = qonnx_model.transform(QCDQToQuant())
qonnx_model.set_opset_import("qonnx.custom_op.general", 1)
qonnx_model = cleanup_model(qonnx_model)
qonnx_model.save(str(qonnx_out))
print("Conversion QONNX terminée :", qonnx_out)
