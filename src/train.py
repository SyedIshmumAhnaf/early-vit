import argparse, torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.seed import set_seed
from src.data.dummy_video import make_loader as make_dummy
from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.losses.earliness import bce_with_earliness
from src.metrics.classification import ap_auc
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data.dada import make_dada_loader
from src.models.fixation_head import FixationHead

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dummy", choices=["dummy","dada"])
    ap.add_argument("--dada_root", type=str, default="")  # set when dataset=dada
    ap.add_argument("--split", type=str, default="training", choices=["training","validation","testing"])

    ap.add_argument("--backbone", type=str, default="tiny3d", choices=["tiny3d","mvitv2_s"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)

    ap.add_argument("--train_n", type=int, default=64)  # only used by dummy
    ap.add_argument("--val_n", type=int, default=32)    # only used by dummy
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_items", type=int, default=0)  # 0=all; useful for quick sanity on DADA

    ap.add_argument("--use_fixation_head", action="store_true")
    ap.add_argument("--fix_lambda", type=float, default=0.1)
    ap.add_argument("--fix_size", type=int, default=14)
    ap.add_argument("--fix_sigma", type=float, default=2.0)
    return ap.parse_args()

def build_loaders(args):
    if args.dataset == "dummy":
        train_loader = make_dummy(batch_size=args.batch, n_samples=args.train_n,
                                  frames=args.frames, size=(args.size,args.size))
        val_loader = make_dummy(batch_size=args.batch, n_samples=args.val_n,
                                frames=args.frames, size=(args.size,args.size), shuffle=False)
        return train_loader, val_loader

    # DADA path
    if not args.dada_root or not os.path.isdir(args.dada_root):
        raise SystemExit("Please provide --dada_root pointing to DADA-2000-small directory.")

    max_items = None if args.max_items <= 0 else args.max_items
    train_loader = make_dada_loader(root_dir=args.dada_root, split="training",
                                    batch=args.batch, frames=args.frames, size=(args.size,args.size),
                                    shuffle=True, max_items=max_items,
                                    use_fixation=args.use_fixation_head, fix_size=args.fix_size, sigma=args.fix_sigma, num_workers=2)
    val_loader   = make_dada_loader(root_dir=args.dada_root, split="validation",
                                    batch=args.batch, frames=args.frames, size=(args.size,args.size),
                                    shuffle=False, max_items=max_items,
                                    use_fixation=args.use_fixation_head, fix_size=args.fix_size, sigma=args.fix_sigma, num_workers=2)
    return train_loader, val_loader

def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_loaders(args)

    # Model
    D = 256
    bb = build_backbone("mvitv2_s" if (device=="cuda" and args.backbone=="mvitv2_s") else "tiny3d", out_dim=D).to(device)
    cls_head = BCEHead(D).to(device) #new
    fix_head = FixationHead(in_dim=D, hm_size=args.fix_size).to(device) if args.use_fixation_head else None #new

    # A tiny wrapper so we can .train() / .eval() in one call
    class Net(torch.nn.Module):
        def __init__(self, bb, cls_head, fix_head=None):
            super().__init__()
            self.bb = bb
            self.cls_head = cls_head
            self.fix_head = fix_head
        def forward(self, x):
            feats = self.bb(x)                  # (B,D)
            logits = self.cls_head(feats)       # (B,)
            logp_map = None
            if self.fix_head is not None:
                logp_map = self.fix_head(feats) # (B,1,Hm,Wm) log-probs
            return logits, logp_map
        
    net = Net(bb, cls_head, fix_head).to(device)
    
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr
    )
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    kldiv = torch.nn.KLDivLoss(reduction="batchmean") if args.use_fixation_head else None
    scaler = amp.GradScaler("cuda", enabled=(device=="cuda"))
    best_ap = -1.0
    
    for epoch in range(1, args.epochs+1):
        net.train()
        losses = []
        for xb, *rest in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            if args.use_fixation_head:
                yb, tb, fixb = rest
            else:
                yb, tb = rest
                fixb = None
            
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=(device=="cuda")):
                logits, logp_map = net(xb)
                loss_cls, w = bce_with_earliness(logits, yb, tb, T=args.frames, pos_weight=1.5)
                loss = loss_cls
                if args.use_fixation_head:
                    tgt = torch.stack(
                        [f if f is not None else torch.full((1,args.fix_size,args.fix_size),
                                                            1.0/(args.fix_size*args.fix_size))
                        for f in fixb],
                        dim=0
                    ).to(device)
                    tgt = tgt / (tgt.sum(dim=(2,3), keepdim=True) + 1e-8)
                    loss_fix = kldiv(logp_map, tgt)
                    loss = loss + args.fix_lambda * loss_fix
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        print(f"Train loss: {sum(losses)/len(losses):.4f}")
        if args.use_fixation_head:
            # Print once per epoch (first batch values) for sanity:
            print(f"(debug) last batch: loss_cls={loss_cls.item():.4f}, loss_fix={loss_fix.item():.4f}")

        # validation
        net.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, *rest in val_loader:
                yb, tb = rest[:2]  # ignore fixation in val metrics
                xb = xb.to(device)
                with amp.autocast("cuda", enabled=(device=="cuda")):
                    logits, _ = net(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist())
                labels.extend(yb.numpy().tolist())
        ap, auc = ap_auc(preds, labels)
        print(f"Val AP: {ap:.3f}  AUC: {auc:.3f}")
        scheduler.step()

        # ... after print(f"Val AP: {ap:.3f}  AUC: {auc:.3f}")
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "bb": net.bb.state_dict(),
            "cls_head": net.cls_head.state_dict(),
            "fix_head": (net.fix_head.state_dict() if net.fix_head is not None else None),
            "args": vars(args),
            "epoch": epoch
        }
        torch.save(ckpt, "checkpoints/last.pt")

        flat = torch.nn.Sequential(net.bb, net.cls_head).state_dict()
        torch.save(flat, "checkpoints/last_flat.pt")

        if ap > best_ap:
            best_ap = ap
            torch.save(ckpt, "checkpoints/best.pt")
            torch.save(flat, "checkpoints/best_flat.pt")
            print(f"↑ New best AP {best_ap:.3f} — saved to checkpoints/best.pt & best_flat.pt")

if __name__ == "__main__":
    main()
