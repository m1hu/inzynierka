import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layers(x)

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DoubleConv(3, 8)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bottleneck = DoubleConv(8, 16)
        self.upconv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder = DoubleConv(16, 8)
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        skip_connection1 = self.encoder(x)
        bottleneck = self.bottleneck(self.pool(skip_connection1))
        upsampled = self.upconv(bottleneck)

        combined = torch.cat((upsampled, skip_connection1), dim=1)
        logits = self.head(self.decoder(combined))

        return logits