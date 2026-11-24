"""Construction de YOLOv8m quantifié (Brevitas), partagée par yolov8m_quant.py et export_yolov8m_onnx.py."""
import torch
import brevitas.nn as qnn
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int32Bias
from ultralytics import YOLO

from parameters import CHECKPOINTS_DIR

FP32_WEIGHTS = CHECKPOINTS_DIR / "yolov8m.pt"


def ptq_checkpoint(bits):
    return CHECKPOINTS_DIR / f"yolov8m_int{bits}_ptq.pth"


def build_quant_yolo(bits):
    """YOLO pré-entraîné, Conv+BN fusionnées, chaque Conv2d -> QuantConv2d INT<bits> (scales en puissance de 2).

    Les scales d'activation ne sont pas encore calibrées : calibrer, ou charger ptq_checkpoint(bits).
    """
    yolo = YOLO(FP32_WEIGHTS)
    net = yolo.model.fuse(verbose=False).eval()  # sinon BN reste en float
    conv_map = {
        torch.nn.Conv2d: (qnn.QuantConv2d, {
            "input_quant": Int8ActPerTensorFixedPoint, "input_bit_width": bits,
            "weight_quant": Int8WeightPerTensorFixedPoint, "weight_bit_width": bits,
            "bias_quant": Int32Bias, "return_quant_tensor": False}),
    }
    # La conv DFL (poids fixes 0..15) fait partie du post-traitement : on la laisse en float.
    yolo.model = layerwise_quantize(net, compute_layer_map=conv_map, name_blacklist=["dfl"])
    return yolo
