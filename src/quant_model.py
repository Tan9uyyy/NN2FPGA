import torch.nn as nn
import brevitas.nn as qnn


class QuantCNN(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(1, 8, 3, padding=1, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            nn.MaxPool2d(2),
            qnn.QuantConv2d(8, 16, 3, padding=1, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            nn.MaxPool2d(2),
        )
        self.classifier = qnn.QuantLinear(16 * 7 * 7, 10, weight_bit_width=bit_width)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
