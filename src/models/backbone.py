import torch
import torch.nn as nn

class Tiny3DConv(nn.Module):
    """Fallback 3D CNN: (B,3,T,H,W) -> (B, D)"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):  # x: (B,3,T,H,W)
        h = self.net(x)           # (B,64,1,1,1)
        h = h.flatten(1)          # (B,64)
        return self.proj(h)       # (B,out_dim)

def build_backbone(name: str = "mvitv2_s", out_dim: int = 256):
    """
    name: 'mvitv2_s' or 'tiny3d'
    """
    if name == "tiny3d":
        return Tiny3DConv(out_dim=out_dim)

    try:
        import timm
        # timm model expects (B,3,T,H,W) when is_video_model=True
        model = timm.create_model("mvitv2_small", pretrained=True, num_classes=0,  # feature extractor
                                  in_chans=3, drop_path_rate=0.1, dynamic_img_size=True)
        # Many timm video models output a feature vec directly
        feat_dim = model.num_features if hasattr(model, "num_features") else 768
        # Wrap to ensure (B,3,T,H,W) -> (B,out_dim)
        class Wrapper(nn.Module):
            def __init__(self, m, feat_dim, out_dim):
                super().__init__()
                self.m = m
                self.proj = nn.Linear(feat_dim, out_dim)
            def forward(self, x):
                # Some timm video models accept (B,3,T,H,W); if not, reshape can be added later.
                feats = self.m(x)
                return self.proj(feats)
        return Wrapper(model, feat_dim, out_dim)
    except Exception:
        # Fallback for local CPU env without timm/video weights
        return Tiny3DConv(out_dim=out_dim)
