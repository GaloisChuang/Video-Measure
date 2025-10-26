import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def parse_classes_arg(arg, model_names):
    """
    Parse --classes which can be:
      - comma-separated names, e.g. "person,dog"
      - comma-separated ids,   e.g. "0,1,2"
      - empty/None => use all classes
    Returns a list of numeric class IDs or None.
    """
    if arg is None or arg.strip() == "":
        return None

    parts = [p.strip() for p in arg.split(",")]
    ids = []
    all_names_to_id = {name: i for i, name in model_names.items()}
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            if p in all_names_to_id:
                ids.append(all_names_to_id[p])
            else:
                raise ValueError(f"Unknown class name: {p}. Known names: {list(all_names_to_id.keys())}")
    return ids


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-instance binary masks from a video with CONSISTENT IDs using YOLOv8-seg tracking."
    )
    parser.add_argument("video", help="Path to the input video.")
    parser.add_argument("out_dir", help="Output directory to write masks: instance_xxx/frame_xxxxxx.png")
    parser.add_argument("--model", default="yolov8l-seg.pt", help="YOLOv8 segmentation model weights.")
    parser.add_argument("--device", default=None, help="Device: 'cpu', 'cuda', or 'cuda:0' etc. (auto if omitted)")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size (pixels).")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--classes", type=str, default=None,
                        help="Filter by classes (comma names or ids). Example: 'person,dog' or '0,1'.")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Optionally cap the number of tracked instances saved.")
    parser.add_argument("--binary_threshold", type=float, default=0.5,
                        help="Threshold to binarize masks (0..1 -> 0/255).")
    parser.add_argument("--save_vis", action="store_true",
                        help="Also save a visualization video with colored masks overlaid.")
    parser.add_argument("--vis_out", default="overlay.mp4", help="Path for visualization video.")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    # Load model
    model = YOLO(args.model)

    # Resolve class filter
    class_ids = parse_classes_arg(args.classes, model.names)  # None or list[int]

    # Tracker ID -> compact 0..K mapping so folders are neat
    id_remap = {}    # tracker_id -> compact_id
    next_compact = 0

    # For overlay video
    vis_writer = None
    vis_fps = None
    frame_h = frame_w = None

    frame_idx = 0
    print("Starting tracking/segmentation... (consistent IDs via persist=True)")

    # stream=True yields Results per-frame; persist=True keeps tracking IDs consistent
    for res in model.track(
        source=args.video,
        stream=True,
        persist=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=class_ids,
        verbose=False
    ):
        # res.orig_img: original frame (H,W,C) BGR
        frame = res.orig_img
        if frame is None:
            # Some versions may not set orig_img; fallback: skip
            frame_idx += 1
            continue

        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]

        # Prepare overlay writer if requested
        if args.save_vis and vis_writer is None:
            # Try to read FPS from source video
            cap = cv2.VideoCapture(args.video)
            if cap.isOpened():
                vis_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            else:
                vis_fps = 30.0
            vis_writer = cv2.VideoWriter(
                args.vis_out,
                cv2.VideoWriter_fourcc(*"mp4v"),
                vis_fps,
                (frame_w, frame_h)
            )

        # If no masks in this frame, optionally still write a blank visual frame
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            if args.save_vis and vis_writer is not None:
                vis_writer.write(frame.copy())
            frame_idx += 1
            continue

        masks = res.masks.data.cpu().numpy()  # (N, H, W) in [0,1]
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # (N,)
        ids = res.boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().astype(int)  # tracker IDs, stable across frames
        else:
            # Fallback: no tracking IDs available (shouldn't happen with persist=True).
            # Use per-frame indices (not consistent). We warn and proceed.
            print("[WARN] No tracker IDs found; falling back to per-frame indices (inconsistent).")
            ids = np.arange(len(masks), dtype=int)

        # Optionally filter by classes again (defensive)
        keep = np.ones(len(masks), dtype=bool)
        if class_ids is not None:
            keep &= np.isin(cls_ids, class_ids)

        # Apply max_instances if set
        if args.max_instances is not None and args.max_instances < keep.sum():
            # Keep highest-confidence first if available (res.boxes.conf)
            confs = res.boxes.conf.cpu().numpy()
            kept_indices = np.where(keep)[0]
            order = kept_indices[np.argsort(-confs[kept_indices])[:args.max_instances]]
            mask_indices = set(order.tolist())
            keep = np.array([i in mask_indices for i in range(len(masks))], dtype=bool)

        # Prepare overlay copy
        vis_frame = frame.copy()

        for k in np.where(keep)[0]:
            tracker_id = int(ids[k])

            # Remap tracker id to compact contiguous index
            if tracker_id not in id_remap:
                id_remap[tracker_id] = next_compact
                next_compact += 1
                print(f"[ID MAP] tracker {tracker_id} -> instance_{id_remap[tracker_id]:03d}")

            compact_id = id_remap[tracker_id]

            # Binarize mask
            m = (masks[k] > args.binary_threshold).astype(np.uint8) * 255

            # Save mask to instance folder
            inst_dir = os.path.join(args.out_dir, f"instance_{compact_id:03d}")
            ensure_dir(inst_dir)
            out_path = os.path.join(inst_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, m)

            # Optional overlay visualization
            if args.save_vis:
                color = _id_color(compact_id)
                colored = np.zeros_like(frame, dtype=np.uint8)
                colored[m > 0] = color
                vis_frame = cv2.addWeighted(vis_frame, 1.0, colored, 0.4, 0)

                # Draw a small label
                yy, xx = np.where(m > 0)
                if len(yy) > 0:
                    y1, x1 = int(np.median(yy)), int(np.median(xx))
                    cv2.putText(vis_frame, f"id {compact_id}", (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if args.save_vis and vis_writer is not None:
            vis_writer.write(vis_frame)

        frame_idx += 1

    if vis_writer is not None:
        vis_writer.release()

    print("\nDone. Masks saved under:", args.out_dir)
    print("Instance folders use **consistent IDs** across frames (thanks to track+persist).")


def _id_color(idx):
    """
    Generate a visually distinct BGR color for a given instance id (deterministic).
    """
    # simple hash -> color
    rng = (idx * 2654435761) & 0xFFFFFFFF
    r = 50 + (rng & 0xFF) % 206
    g = 50 + ((rng >> 8) & 0xFF) % 206
    b = 50 + ((rng >> 16) & 0xFF) % 206
    return int(b), int(g), int(r)


if __name__ == "__main__":
    main()