import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from tqdm import tqdm
from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.data.dada import key_to_coord_path, first_event_frame_from_coord, _resolve_video_path
from src.utils.seed import set_seed

def read_video_TCHW(path, H, W):
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
    frames = frames.float()/255.
    frames = torch.nn.functional.interpolate(
        frames.permute(1,0,2,3).unsqueeze(0),
        size=(frames.shape[0],H,W),mode="trilinear",align_corners=False
    ).squeeze(0).permute(1,0,2,3)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (frames-mean)/std

def sliding_windows_T(frames_tchw, T):
    outs=[]
    for i in range(frames_tchw.shape[0]):
        s=max(0,i-T+1)
        w=frames_tchw[s:i+1]
        if w.shape[0]<T:
            pad=w[0:1].repeat(T-w.shape[0],1,1,1)
            w=torch.cat([pad,w],dim=0)
        outs.append(w.permute(1,0,2,3))
    return outs

def load_model(ckpt, backbone="mvitv2_s", device="cuda"):
    bb=build_backbone(backbone, out_dim=256).to(device).eval()
    head=BCEHead(256).to(device).eval()
    model=torch.nn.Sequential(bb,head).to(device)
    sd=torch.load(ckpt,map_location=device)
    if "bb" in sd and "cls_head" in sd:
        bb.load_state_dict(sd["bb"],strict=False)
        head.load_state_dict(sd["cls_head"],strict=False)
    else:
        model.load_state_dict(sd,strict=False)
    model.eval()
    return model

def predict_seq(model, frames_tchw, device, T=16, batch_size=16):
    wins=sliding_windows_T(frames_tchw,T)
    probs=[]
    for i in range(0,len(wins),batch_size):
        b=torch.stack(wins[i:i+batch_size]).to(device)
        with torch.no_grad():
            p=torch.sigmoid(model(b)).cpu().numpy().flatten()
        probs.extend(p)
    return np.array(probs)

def visualize_prob_curve(probs_fix, probs_nofix, event_idx, key, save_dir):
    plt.figure(figsize=(10,4))
    plt.plot(probs_nofix,label="No Fixation",color="royalblue")
    plt.plot(probs_fix,label="With Fixation",color="green")
    plt.axvline(event_idx,color="red",linestyle="--",label="Event Frame")
    plt.xlabel("Frame index")
    plt.ylabel("Accident probability")
    plt.title(f"{key} — Probability Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"{key.replace('/','_')}.png"))
    plt.close()

def load_samples(samples_file):
    samples = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 2:
                continue
            key, split = parts
            samples.append((key, split))
    return samples

def main():
    # === configure ===
    #dada_root = "/content/drive/MyDrive/DADA-2000-small"
    dada_root = "/content/DADA-2000-small"
    ckpt_fix = "checkpoints/best_flat_fix.pt"
    ckpt_nofix = "checkpoints/best_flat.pt"
    out_dir = "results/plots"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    samples = load_samples("samples_phase4c.txt")
    print(f"[viz] loaded {len(samples)} samples for phase 4c")

    model_fix = load_model(ckpt_fix, device=device)
    model_nofix = load_model(ckpt_nofix, device=device)

    for key,split in tqdm(samples):
        rgb_root = os.path.join(dada_root, split, "rgb_videos")
        video_path = _resolve_video_path(rgb_root, key)
        if video_path is None:
            print(f"[viz] WARNING: no video file found for key={key}, split={split}")
            continue
        
        frames_tchw = read_video_TCHW(video_path,112,112)
        probs_fix   = predict_seq(model_fix,frames_tchw,device,T=16)
        probs_nofix = predict_seq(model_nofix,frames_tchw,device,T=16)
        coord_path  = key_to_coord_path(dada_root,split,key)
        event_idx   = first_event_frame_from_coord(coord_path)
        visualize_prob_curve(probs_fix,probs_nofix,event_idx,key,out_dir)

if __name__ == "__main__":
    main()
