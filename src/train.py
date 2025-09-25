import argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.seed import set_seed
from src.data.dummy_video import make_loader
from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.losses.earliness import bce_with_earliness
from src.metrics.classification import ap_auc

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="tiny3d", choices=["tiny3d","mvitv2_s"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--train_n", type=int, default=64)
    ap.add_argument("--val_n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(batch_size=args.batch, n_samples=args.train_n,
                               frames=args.frames, size=(args.size,args.size))
    val_loader   = make_loader(batch_size=args.batch, n_samples=args.val_n,
                               frames=args.frames, size=(args.size,args.size), shuffle=False)

    # Model
    D = 256
    bb = build_backbone("mvitv2_s" if (device=="cuda" and args.backbone=="mvitv2_s") else "tiny3d", out_dim=D)
    head = BCEHead(D)
    model = torch.nn.Sequential(bb, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb, tb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, w = bce_with_earliness(logits, yb, tb, T=args.frames, pos_weight=1.5)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Train loss: {sum(losses)/len(losses):.4f}")

        # validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist())
                labels.extend(yb.numpy().tolist())
        ap, auc = ap_auc(preds, labels)
        print(f"Val AP: {ap:.3f}  AUC: {auc:.3f}")

if __name__ == "__main__":
    main()
