from pathlib import Path

import torch
from brevitas.export import export_onnx_qcdq
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.util.cleanup import cleanup_model

from quant_model import QuantCNN

BITS = 8
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / f"mnist_int{BITS}.pth"
OUTPUT = ROOT / "models" / f"mnist_int{BITS}.onnx"
QONNX_OUTPUT = ROOT / "models" / f"mnist_int{BITS}_qonnx.onnx"

model = QuantCNN(bit_width=BITS)
saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
if "state_dict" in saved and saved["bits"] != BITS:
    raise ValueError("Les bits ne correspondent pas au checkpoint.")
model.load_state_dict(saved.get("state_dict", saved))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
export_onnx_qcdq(
    model, dummy_input, export_path=str(OUTPUT), dynamo=False,
    keep_initializers_as_inputs=False,
)
print("Export terminé :", OUTPUT)

# ONNX (QCDQ) -> opérateurs Quant de QONNX.
qonnx_model = cleanup_model(ModelWrapper(str(OUTPUT)))
qonnx_model = qonnx_model.transform(QCDQToQuant())
qonnx_model.set_opset_import("qonnx.custom_op.general", 1)
qonnx_model = cleanup_model(qonnx_model)
qonnx_model.save(str(QONNX_OUTPUT))
print("Conversion QONNX terminée :", QONNX_OUTPUT)
