import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive Conv→BN→ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool then ConvBlock — encoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + skip connection + ConvBlock — decoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SeismicEncoder(nn.Module):
    """Compress seismic (5, 1000, 70) → feature map (C, 70, 70)."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.compress = nn.Sequential(
            # (5, 1000, 70) → (32, 500, 70)
            nn.Conv2d(5, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 500, 70) → (32, 250, 70)
            nn.Conv2d(32, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 250, 70) → (32, 125, 70)
            nn.Conv2d(32, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 125, 70) → (32, 62, 70)
            nn.Conv2d(32, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 62, 70) → (32, 70, 70) via interpolation-free strided conv
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final adaptive pool runs on CPU to avoid MPS bug
        self.proj = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.compress(x)
        x = torch.nn.functional.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)
        x = self.proj(x)
        return x

class WaveformUNet(nn.Module):
    """
    UNet for Full Waveform Inversion.
    Input:  (B, 5, 1000, 70)  seismic waveforms
    Output: (B, 1, 70, 70)    predicted velocity map
    """
    def __init__(self, base_ch=64):
        super().__init__()

        self.encoder    = SeismicEncoder(out_channels=base_ch)

        self.inc        = ConvBlock(base_ch,     base_ch)
        self.down1      = DownBlock(base_ch,     base_ch*2)
        self.down2      = DownBlock(base_ch*2,   base_ch*4)
        self.down3      = DownBlock(base_ch*4,   base_ch*8)

        self.bottleneck = ConvBlock(base_ch*8,   base_ch*8)

        self.up3        = UpBlock(base_ch*8,     base_ch*4)
        self.up2        = UpBlock(base_ch*4,     base_ch*2)
        self.up1        = UpBlock(base_ch*2,     base_ch)

        self.out_conv   = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        x  = self.encoder(x)

        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)

        b  = self.bottleneck(s4)

        x  = self.up3(b,  s3)
        x  = self.up2(x,  s2)
        x  = self.up1(x,  s1)

        x  = self.out_conv(x)
        return self.sigmoid(x)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable