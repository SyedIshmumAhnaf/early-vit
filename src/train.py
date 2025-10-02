import argparse, torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.seed import set_seed
from src.data.dummy_video import make_loader as make_dummy
from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.losses.earliness import bce_with_earliness
from src.metrics.classification import ap_auc
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    return ap.parse_args()

def build_loaders(args):
    if args.dataset == "dummy":
        train_loader = make_dummy(batch_size=args.batch, n_samples=args.train_n,
                                  frames=args.frames, size=(args.size,args.size))
        val_loader = make_dummy(batch_size=args.batch, n_samples=args.val_n,
                                frames=args.frames, size=(args.size,args.size), shuffle=False)
        return train_loader, val_loader

    # DADA path
    from src.data.dada import make_dada_loader
    if not args.dada_root or not os.path.isdir(args.dada_root):
        raise SystemExit("Please provide --dada_root pointing to DADA-2000-small directory.")

    max_items = None if args.max_items <= 0 else args.max_items
    train_loader = make_dada_loader(root_dir=args.dada_root, split="training",
                                    batch=args.batch, frames=args.frames, size=(args.size,args.size),
                                    shuffle=True, max_items=max_items)
    val_loader   = make_dada_loader(root_dir=args.dada_root, split="validation",
                                    batch=args.batch, frames=args.frames, size=(args.size,args.size),
                                    shuffle=False, max_items=max_items)
    return train_loader, val_loader

def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_loaders(args)

    # Model
    D = 256
    bb = build_backbone("mvitv2_s" if (device=="cuda" and args.backbone=="mvitv2_s") else "tiny3d", out_dim=D)
    head = BCEHead(D)
    model = torch.nn.Sequential(bb, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)

    scaler = GradScaler(enabled=(device=="cuda"))
    best_ap = -1.0
    
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb, tb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device=="cuda")):
                logits = model(xb)
                loss, w = bce_with_earliness(logits, yb, tb, T=args.frames, pos_weight=1.5)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        print(f"Train loss: {sum(losses)/len(losses):.4f}")

        # validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                xb = xb.to(device)
                with autocast(enabled=(device=="cuda")):
                    logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist())
                labels.extend(yb.numpy().tolist())
        ap, auc = ap_auc(preds, labels)
        print(f"Val AP: {ap:.3f}  AUC: {auc:.3f}")
        scheduler.step()

        # ... after print(f"Val AP: {ap:.3f}  AUC: {auc:.3f}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/last.pt")
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f"↑ New best AP {best_ap:.3f} — saved to checkpoints/best.pt")


if __name__ == "__main__":
    main()
