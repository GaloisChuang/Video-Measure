"""
ICLR 2025 paper metrics (arXiv 2310.05922v3)
- CLIP-F  (framewise cosine with text)
- CLIP-T  (mean-pooled video cosine with text)
- Warp-Err (TV-L1 optical flow warp MAE; lower is better)
- Q-edit   (CLIP-T(edited) - CLIP-T(original))
"""

import argparse, os, glob, csv, sys, time
import numpy as np
import cv2
import torch
import clip
from typing import List
from PIL import Image

# ----------------- helpers -----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def log(msg: str):
    print(msg, flush=True)

def is_img(p): return os.path.splitext(p)[1].lower() in IMG_EXTS

def read_video(path: str, max_frames=128, uniform=True) -> List[np.ndarray]:
    frames = []
    if os.path.isdir(path):
        files = sorted([f for f in glob.glob(os.path.join(path, "*")) if is_img(f)])
        if not files:
            raise FileNotFoundError(f"No image frames in directory: {path}")
        if uniform and len(files) > max_frames:
            idx = np.linspace(0, len(files)-1, max_frames).round().astype(int)
            files = [files[i] for i in idx]
        else:
            files = files[:max_frames]
        for f in files:
            bgr = cv2.imread(f, cv2.IMREAD_COLOR)
            if bgr is None: continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total > 0:
            idxs = np.linspace(0, total-1, min(max_frames, total)).round().astype(int)
            idxset = set(int(i) for i in idxs)
        else:
            # unknown length, just read up to max_frames
            idxset = None
        cur = 0
        while True:
            ret, bgr = cap.read()
            if not ret: break
            if idxset is None or cur in idxset:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if idxset is None and len(frames) >= max_frames:
                    break
            cur += 1
        cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Zero frames read from: {path}")
    return frames

@torch.no_grad()
def clip_feats(frames, model, preprocess, device, bs=16):
    imgs = [preprocess(Image.fromarray(f)) for f in frames]
    x = torch.stack(imgs).to(device)
    feats = []
    for i in range(0, len(x), bs):
        f = model.encode_image(x[i:i+bs])
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats)

@torch.no_grad()
def text_feat(text, model, device):
    toks = clip.tokenize([text]).to(device)
    e = model.encode_text(toks)
    e = e / e.norm(dim=-1, keepdim=True)
    return e[0]

def compute_clip_ft(frames, text, model, preprocess, device):
    imf = clip_feats(frames, model, preprocess, device)
    tf = text_feat(text, model, device)
    clip_f = (imf @ tf).mean().item()
    vfeat = imf.mean(dim=0)
    vfeat = vfeat / vfeat.norm()
    clip_t = torch.dot(vfeat, tf).item()
    return clip_f, clip_t

def warp_err(frames):
    if len(frames) < 2: return 0.0
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        raise ImportError("cv2.optflow.DualTVL1OpticalFlow_create() not found. Install opencv-contrib-python and ensure opencv-python is uninstalled.")
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    errs = []
    for i in range(len(frames)-1):
        p, n = frames[i], frames[i+1]
        g1 = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(n, cv2.COLOR_RGB2GRAY)
        flow = tvl1.calc(g1, g2, None)
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0].astype(np.float32)
        map_y = grid_y + flow[..., 1].astype(np.float32)
        warped = cv2.remap(p.astype(np.float32)/255.0, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mae = np.mean(np.abs(warped - n.astype(np.float32)/255.0))
        errs.append(mae)
    return float(np.mean(errs))

def q_edit(edited_frames, orig_frames, text, model, preprocess, device):
    _, ct_edit = compute_clip_ft(edited_frames, text, model, preprocess, device)
    _, ct_orig = compute_clip_ft(orig_frames, text, model, preprocess, device)
    return float(ct_edit - ct_orig)

def main():
    ap = argparse.ArgumentParser(description="ICLR 2025 metrics: CLIP-F / CLIP-T / Warp-Err / Q-edit")
    ap.add_argument("--video", required=True, help="Edited video or frame dir")
    ap.add_argument("--text", required=True, help="Prompt text / edit instruction")
    ap.add_argument("--orig", help="Original video or frame dir (for Q-edit)")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--out_csv", default="metrics.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip", default="ViT-L/14", help="CLIP backbone (e.g., ViT-L/14)")
    args = ap.parse_args()

    log("==> Starting evaluation")
    log(f"Device: {args.device} | CLIP: {args.clip}")

    # Load CLIP (warn if it might need to download)
    try:
        log("Loading CLIP…")
        t0 = time.time()
        model, preprocess = clip.load(args.clip, device=args.device)
        log(f"CLIP loaded in {time.time()-t0:.2f}s")
    except Exception as e:
        log(f"ERROR loading CLIP weights: {e}")
        log("If this is a fresh environment, CLIP may need internet to download weights once.")
        sys.exit(1)

    # Read videos
    log(f"Reading edited video/frames: {args.video}")
    edited = read_video(args.video, args.max_frames)
    log(f"Edited frames: {len(edited)}")

    orig = None
    if args.orig:
        log(f"Reading original video/frames: {args.orig}")
        orig = read_video(args.orig, args.max_frames)
        log(f"Original frames: {len(orig)}")

    # Compute metrics
    log("Computing CLIP-F / CLIP-T …")
    clip_f, clip_t = compute_clip_ft(edited, args.text, model, preprocess, args.device)
    log("Computing Warp-Err (TV-L1) …")
    warp = warp_err(edited)
    q = None
    if orig is not None:
        log("Computing Q-edit …")
        q = q_edit(edited, orig, args.text, model, preprocess, args.device)

    # Report
    log("\n=== Metrics ===")
    log(f"CLIP-F   : {clip_f:.5f}")
    log(f"CLIP-T   : {clip_t:.5f}")
    log(f"Warp-Err : {warp:.5f}  (lower is better)")
    if q is not None:
        log(f"Q-edit   : {q:.5f}  (CLIP-T(edited) - CLIP-T(original))")
    else:
        log("Q-edit   : (provide --orig to compute)")
    log("================")

    # Save CSV (no pandas)
    file_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video","clip_f","clip_t","warp_err","q_edit"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "video": args.video,
            "clip_f": f"{clip_f:.8f}",
            "clip_t": f"{clip_t:.8f}",
            "warp_err": f"{warp:.8f}",
            "q_edit": "" if q is None else f"{q:.8f}",
        })
    log(f"Saved metrics to {args.out_csv}")

if __name__ == "__main__":
    main()


'''
Usage:
python Measure.py --video <EDITED_VIDEO> --text "<TEXT_PROMPT>" [--orig <ORIGINAL_VIDEO>] [--device cuda|cpu]
