import argparse
from pathlib import Path

import torch
from brevitas.export import export_onnx_qcdq
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.util.cleanup import cleanup_model

from quant_model import QuantCNN

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=range(2, 33), default=8, help="Nombre de bits (par défaut : 8).")
args = parser.parse_args()
bits = args.bits

ROOT = Path(__file__).resolve().parent.parent
checkpoint = ROOT / "checkpoints" / f"mnist_int{bits}.pth"
if not checkpoint.is_file():
    parser.error(f"Poids absents : entraîne d'abord le modèle INT{bits}.")

output = ROOT / "models" / f"mnist_int{bits}.onnx"
qonnx_output = ROOT / "models" / f"mnist_int{bits}_qonnx.onnx"

model = QuantCNN(bit_width=bits)
saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
if "state_dict" in saved and saved.get("bits") != bits:
    parser.error("Le nombre de bits ne correspond pas aux poids sauvegardés.")
model.load_state_dict(saved.get("state_dict", saved))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
output.parent.mkdir(parents=True, exist_ok=True)
export_onnx_qcdq(
    model, dummy_input, export_path=str(output), dynamo=False,
    keep_initializers_as_inputs=False,
)
print("Export terminé :", output)

# ONNX (QCDQ) -> opérateurs Quant de QONNX.
qonnx_model = cleanup_model(ModelWrapper(str(output)))
qonnx_model = qonnx_model.transform(QCDQToQuant())
qonnx_model.set_opset_import("qonnx.custom_op.general", 1)
qonnx_model = cleanup_model(qonnx_model)
qonnx_model.save(str(qonnx_output))
print("Conversion QONNX terminée :", qonnx_output)
