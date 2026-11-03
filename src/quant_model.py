import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int32Bias, Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Uint8ActPerTensorFixedPoint


class QuantCNN(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True,
                                act_quant=Int8ActPerTensorFixedPoint),
            qnn.QuantConv2d(1, 8, 3, padding=1, weight_bit_width=bit_width,
                                bias_quant=Int32Bias,
                                weight_quant=Int8WeightPerTensorFixedPoint),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True,
                                act_quant=Uint8ActPerTensorFixedPoint),
            nn.MaxPool2d(2),
            qnn.QuantConv2d(8, 16, 3, padding=1, weight_bit_width=bit_width,
                                bias_quant=Int32Bias,
                                weight_quant=Int8WeightPerTensorFixedPoint),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True,
                                act_quant=Uint8ActPerTensorFixedPoint),
            nn.MaxPool2d(2),
        )
        self.classifier = qnn.QuantConv2d(16, 10, 7, weight_bit_width=bit_width, bias_quant=Int32Bias,
                                output_quant=Int8ActPerTensorFixedPoint,
                                weight_quant=Int8WeightPerTensorFixedPoint)

    def forward(self, x):
        return self.classifier(self.features(x)).flatten(1)
