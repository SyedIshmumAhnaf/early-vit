import torch

def adalea_weights(tte, T, alpha=0.7):
    """
    Simple earliness weight: earlier = heavier.
    tte: (B,) integer time-to-event in frames (1..T); negatives can pass any value (we set weight=1.0 on negatives).
    alpha in (0,1): controls steepness.
    """
    # smaller tte => larger weight; normalize to [0,1]
    w = 1.0 - (tte.float() - 1) / (T - 1 + 1e-6)
    # smooth schedule
    return (alpha + (1 - alpha) * w).clamp(0., 1.)

def bce_with_earliness(logits, targets, tte, T, pos_weight=1.0):
    """
    logits: (B,)
    targets: (B,) in {0,1}
    tte: (B,) time-to-event (arbitrary for negatives)
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction='none', pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    # apply earliness weights only to positives
    w = torch.ones_like(bce)
    pos_mask = (targets > 0.5)
    w[pos_mask] = adalea_weights(tte[pos_mask], T)
    loss = (w * bce).mean()
    return loss, w.detach()
