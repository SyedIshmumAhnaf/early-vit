import torch
from src.data.dummy_video import make_loader
from src.models.backbone import build_backbone
from src.models.head import BCEHead

def test_pipeline_shapes():
    dl = make_loader(batch_size=2, n_samples=4, frames=8, size=(64,64))
    xb,yb,tb = next(iter(dl))
    assert xb.shape == (2,3,8,64,64)
    bb = build_backbone("tiny3d", out_dim=128)
    hd = BCEHead(128)
    logits = hd(bb(xb))
    assert logits.shape == (2,)
