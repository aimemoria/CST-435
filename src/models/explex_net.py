import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionCNN(nn.Module):
    """Image branch: ResNet18 trunk → projection."""
    def __init__(self, out_dim: int = 128, train_backbone: bool = False, in_channels: int = 3):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # If input channels != 3, we need to modify the first conv layer
        if in_channels != 3:
            # Create new first conv layer with correct input channels
            original_conv = m.conv1
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize weights by averaging across the extra channels or repeating
            with torch.no_grad():
                if in_channels > 3:
                    # Repeat the original weights across new channels
                    m.conv1.weight[:, :3] = original_conv.weight
                    m.conv1.weight[:, 3:] = original_conv.weight.repeat(1, in_channels // 3, 1, 1)[:, :(in_channels-3)]
                else:
                    m.conv1.weight = original_conv.weight[:, :in_channels]

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)      # [B,512]
        return F.relu(self.proj(h))          # [B,out_dim]

class AudioCNN(nn.Module):
    """Audio branch: small 2D CNN over log-mel spectrograms."""
    def __init__(self, out_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)           # [B,128]
        return F.relu(self.proj(h))          # [B,out_dim]

class ExplexNet(nn.Module):
    """Explosion-or-Explanation classifier (late fusion)."""
    def __init__(self, vid_dim: int = 128, aud_dim: int = 128, vid_in_channels: int = 9):
        super().__init__()
        self.vision = VisionCNN(out_dim=vid_dim, in_channels=vid_in_channels)
        self.audio  = AudioCNN(out_dim=aud_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vid_dim + aud_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # sigmoid
        )

    def forward(self, img: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        zv = self.vision(img)
        za = self.audio(mel)
        z = torch.cat([zv, za], dim=1)
        return self.classifier(z)  # [B,1] (logits)
