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

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2, stride=2)
        self.encoder1 = DoubleConv(3, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(64, 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x) # (B, 32, 256, 256)
        pooled1 = self.pool(skip1) # (B, 32, 128, 128)

        skip2 = self.encoder2(pooled1) # (B, 64, 128, 128)
        pooled2 = self.pool(skip2) # (B, 64, 64, 64)

        skip3 = self.encoder3(pooled2) # (B, 128, 64, 64)
        pooled3 = self.pool(skip3) # (B, 128, 32, 32)

        skip4 = self.encoder4(pooled3) # (B, 256, 32, 32)
        pooled4 = self.pool(skip4) # (B, 256, 16, 16)

        bottleneck = self.bottleneck(pooled4) # (B, 512, 16, 16)

        upsampled1 = self.upconv1(bottleneck) # (B, 256, 32, 32)
        combined1 = torch.cat((upsampled1, skip4), dim=1) # (B, 512, 32, 32)
        dec_conv1 = self.decoder1(combined1) # (B, 256, 32, 32)

        upsampled2 = self.upconv2(dec_conv1) # (B, 128, 64, 64)
        combined2 = torch.cat((upsampled2, skip3), dim=1) # (B, 256, 64, 64)
        dec_conv2 = self.decoder2(combined2) # (B, 128, 64, 64)

        upsampled3 = self.upconv3(dec_conv2) # (B, 64, 128, 128)
        combined3 = torch.cat((upsampled3, skip2), dim=1) # (B, 128, 128, 128)
        dec_conv3 = self.decoder3(combined3) # (B, 64, 128, 128)
        
        upsampled4 = self.upconv4(dec_conv3) # (B, 32, 256, 256)
        combined4 = torch.cat((upsampled4, skip1), dim=1) # (B, 64, 256, 256)
        dec_conv4 = self.decoder4(combined4) # (B, 32, 256, 256)

        logits = self.head(dec_conv4)

        return logits
        
        
        

        

        