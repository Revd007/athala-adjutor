import torch
import torch.nn as nn
from logger import logger

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        try:
            x = self.network(x)
            x = x[:, :, -1]  # Take last time step
            x = self.linear(x)
            return x
        except Exception as e:
            logger.error(f"TCN forward error: {e}")
            raise

if __name__ == "__main__":
    model = TCN(input_size=5, output_size=1, num_channels=[16, 32, 64])
    x = torch.randn(32, 5, 100)  # Batch, channels, sequence
    print(model(x).shape)  # Should be [32, 1]