import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import argparse
import sys
import os
import shutil
import tempfile
import subprocess
import torch
import lpips
import clip
from PIL import Image
from tqdm import tqdm

# This code calculates PSNR, SSIM, LPIPS, CLIP Similarity, and FID between two videos.

def frame_to_tensor_lpips(frame, device):
    """Converts a BGR NumPy frame to a RGB PyTorch tensor for LPIPS."""
    return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1

def calculate_fid_for_videos(video_path1, video_path2, device):
    """
    Calculates the Frechet Inception Distance (FID) between two videos.
    This function extracts all frames from both videos into temporary directories
    and then uses the pytorch-fid command-line tool to compute the score.
    """
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        print(f"\nExtracting frames to temporary directories for FID calculation...")
        
        # Helper to extract frames
        def extract_frames(vid_path, out_dir):
            cap = cv2.VideoCapture(vid_path)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(out_dir, f"frame_{count:06d}.png"), frame)
                count += 1
            cap.release()
            return count

        count1 = extract_frames(video_path1, dir1)
        count2 = extract_frames(video_path2, dir2)
        print(f"Extracted {count1} frames from video 1 and {count2} frames from video 2.")

        if count1 == 0 or count2 == 0:
            print("Error: One of the videos has no frames. Cannot calculate FID.")
            return None

        # --- Run the pytorch-fid command ---
        print("Running FID calculation (this may take a while)...")
        command = [
            sys.executable, "-m", "pytorch_fid",
            dir1,
            dir2,
            "--device", str(device)
        ]
        
        # try:
        #     result = subprocess.run(command, capture_output=True, text=True, check=True)
        #     output = result.stdout
        #     # Parse the output to find the FID score
        #     for line in output.splitlines():
        #         if "Frechet Inception Distance:" in line:
        #             fid_score = float(line.split(":")[1].strip())
        #             return fid_score
        # except subprocess.CalledProcessError as e:
        #     print("\n--- FID Calculation Error ---")
        #     print(f"The 'pytorch-fid' command failed with exit code {e.returncode}.")
        #     print("Please ensure 'pytorch-fid' is installed and works correctly.")
        #     print("Error output:\n", e.stderr)
        #     return None
        # except FileNotFoundError:
        #     print("\nError: Could not find the 'pytorch-fid' package.")
        #     print("Please install it with: pip install pytorch-fid")
        #     return None
        import re

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            out = (result.stdout or "") + "\n" + (result.stderr or "")
        
            # Common formats:
            # "Frechet Inception Distance: 12.3456"
            # "FID: 12.3456"
            # "fid: 12.3456"
            m = re.search(r'(Frechet Inception Distance|FID)\s*:\s*([0-9]*\.?[0-9]+)', out, re.IGNORECASE)
            if m:
                fid_score = float(m.group(2))
                return fid_score
        
            # If return code nonzero, show why
            if result.returncode != 0:
                print("\n--- FID Calculation Error ---")
                print(f"Exit code: {result.returncode}")
                print("STDOUT:\n", result.stdout)
                print("STDERR:\n", result.stderr)
                return None
        
            # Return code 0 but couldn't parse => show raw output for visibility
            print("\n--- FID Calculation: Could not parse score ---")
            print("Raw output:\n", out)
            print("Tip: Update the regex if your pytorch-fid version prints a different label.")
            return None
        
        except FileNotFoundError:
            print("\nError: Could not find the 'pytorch-fid' package.")
            print("Please install it with: pip install pytorch-fid")
            return None
    return None


def run_full_analysis(args):
    """
    Main function to run the complete video analysis based on parsed arguments.
    """
    # --- Model and Device Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    lpips_model = lpips.LPIPS(net=args.net).to(device)
    print(f"LPIPS model ('{args.net}' network) loaded.")

    clip_model, clip_preprocess = (None, None)
    if args.prompt:
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print("CLIP model ('ViT-B/32') loaded.")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Please install it with: pip install git+https://github.com/openai/CLIP.git")
            return

    # --- Video File Handling ---
    original_cap = cv2.VideoCapture(args.original_video)
    processed_cap = cv2.VideoCapture(args.processed_video)

    if not original_cap.isOpened() or not processed_cap.isOpened():
        print("Error opening video files.")
        return

    num_frames = min(int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(processed_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    if num_frames == 0:
        print("Error: One or both videos have 0 frames.")
        return
    
    print(f"\nComparing {num_frames} frames from each video for per-frame metrics...")
    print("-" * 30)

    # --- Per-Frame Metric Calculation Loop ---
    results = []
    text_tokens = clip.tokenize([args.prompt]).to(device) if args.prompt else None

    for frame_num in tqdm(range(num_frames)):
        ret_orig, frame_orig = original_cap.read()
        ret_proc, frame_proc = processed_cap.read()
        if not ret_orig or not ret_proc: break

        # Calculate standard metrics
        psnr_val = psnr(frame_orig, frame_proc, data_range=255)
        ssim_val = ssim(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY), data_range=255)

        current_metrics = {'frame': frame_num, 'psnr': psnr_val, 'ssim': ssim_val}

        # Calculate DL-based metrics
        with torch.no_grad():
            tensor_orig_lpips = frame_to_tensor_lpips(frame_orig, device)
            tensor_proc_lpips = frame_to_tensor_lpips(frame_proc, device)
            lpips_val = lpips_model(tensor_orig_lpips, tensor_proc_lpips).item()
            current_metrics['lpips'] = lpips_val

            if clip_model and text_tokens is not None:
                pil_image = Image.fromarray(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB))
                image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
                logits_per_image, _ = clip_model(image_input, text_tokens)
                clip_sim = logits_per_image.item() / 100.0 # Scale factor is convention
                current_metrics['clip_similarity'] = clip_sim
        
        results.append(current_metrics)
        
        progress_str = f"\rFrame {frame_num+1}/{num_frames} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}"
        if 'clip_similarity' in current_metrics:
            progress_str += f" | CLIP: {current_metrics['clip_similarity']:.4f}"
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    original_cap.release()
    processed_cap.release()
    
    print("\n" + "-" * 30)
    print("Per-frame analysis complete.")
    
    # --- Reporting Per-Frame Results ---
    if results:
        df = pd.DataFrame(results)
        print("\n--- Overall Average Per-Frame Results ---")
        summary = df.mean().drop('frame')
        print(summary.to_string())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nDetailed per-frame report saved to: {args.output}")
    
    # --- FID Calculation (Optional) ---
    if args.calculate_fid:
        fid_score = calculate_fid_for_videos(args.original_video, args.processed_video, device)
        if fid_score is not None:
            print("\n--- Overall Distributional Result ---")
            print(f"Frechet Inception Distance (FID) (Lower is better): {fid_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A comprehensive tool to calculate PSNR, SSIM, LPIPS, CLIP Similarity, and FID between two videos.")
    parser.add_argument("original_video", help="Path to the original (reference) video file.")
    parser.add_argument("processed_video", help="Path to the processed (test) video file.")
    parser.add_argument("-o", "--output", help="Optional: Path to save a CSV file with detailed frame-by-frame results.", default=None)
    parser.add_argument("-n", "--net", help="LPIPS network type to use.", choices=['alex', 'vgg'], default='alex')
    parser.add_argument("-p", "--prompt", help="Optional: Text prompt to calculate CLIP similarity against the processed video.", default=None)
    parser.add_argument("--calculate-fid", action='store_true', help="Optional: Enable FID calculation. This is slow and requires significant disk space for temporary frame extraction.")
    
    args = parser.parse_args()
    run_full_analysis(args)


'''
Usage: python3 PSNR_SSIM.py ./results/xx.mp4 ./goldens/xx.mp4 -o ./results/xx_report.csv -n alex -p "Prompt" --calculate-fid
'''