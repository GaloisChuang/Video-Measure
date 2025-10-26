import cv2
import numpy as np
import torch
import clip
from PIL import Image
import argparse
import sys

# This code analyzes the temporal consistency of a video using two metrics:
# 1. Warping Error: Measures pixel-level consistency using optical flow.
# 2. CLIP Consistency: Measures semantic consistency using CLIP embeddings.

def calculate_warping_error(prev_frame, current_frame):
    """
    Calculates the warping error between two frames using Dense Optical Flow.

    Args:
        prev_frame (np.ndarray): The previous frame (in BGR format).
        current_frame (np.ndarray): The current frame (in BGR format).

    Returns:
        float: The Mean Absolute Error (MAE) between the warped previous frame
               and the current frame. Lower values indicate better consistency.
    """
    # Convert frames to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    # This returns a 2-channel array with the x and y displacement for each pixel
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Create a meshgrid to represent the pixel coordinates of the current frame
    h, w = current_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Create the map for remapping: new_coords = old_coords + flow
    map_x = (x_coords + flow[..., 0]).astype(np.float32)
    map_y = (y_coords + flow[..., 1]).astype(np.float32)

    # Warp the previous frame to predict the current frame based on the flow
    # cv2.remap applies the transformation defined by the maps
    warped_prev_gray = cv2.remap(prev_gray, map_x, map_y, cv2.INTER_LINEAR)

    # Calculate the Mean Absolute Error (L1 norm) between the warped frame and the actual current frame
    warping_error = np.mean(np.abs(current_gray.astype(float) - warped_prev_gray.astype(float)))
    
    return warping_error

def calculate_clip_consistency(prev_frame, current_frame, clip_model, preprocess, device):
    """
    Calculates the cosine similarity of CLIP embeddings for two consecutive frames.

    Args:
        prev_frame (np.ndarray): The previous frame (in BGR format).
        current_frame (np.ndarray): The current frame (in BGR format).
        clip_model: The loaded CLIP model.
        preprocess: The CLIP image preprocessor.
        device (torch.device): The device to run the model on.

    Returns:
        float: The cosine similarity score. Higher values (closer to 1.0)
               indicate better semantic consistency.
    """
    # Preprocess both frames for CLIP
    prev_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
    current_pil = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    
    prev_input = preprocess(prev_pil).unsqueeze(0).to(device)
    current_input = preprocess(current_pil).unsqueeze(0).to(device)

    # Get the image embeddings (features) from CLIP
    with torch.no_grad():
        prev_features = clip_model.encode_image(prev_input)
        current_features = clip_model.encode_image(current_input)

    # Normalize the features
    prev_features /= prev_features.norm(dim=-1, keepdim=True)
    current_features /= current_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (prev_features @ current_features.T).item()
    
    return similarity

def analyze_temporal_consistency(video_path):
    """
    Main function to analyze a video's temporal consistency using both
    warping error and CLIP similarity.
    """
    # --- 1. Initialize models and device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP model ('ViT-B/32') loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Please install it with: pip install git+https://github.com/openai/CLIP.git")
        return

    # --- 2. Open video file ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames < 2:
        print("Error: Video must have at least 2 frames for analysis.")
        return

    print(f"\nAnalyzing {num_frames} frames from '{video_path}'...")
    print("-" * 30)

    # --- 3. Loop through frames and calculate metrics ---
    results = []
    ret, prev_frame = cap.read() # Read the first frame

    for frame_num in range(1, num_frames):
        ret, current_frame = cap.read()
        if not ret:
            break

        # Calculate both metrics for the frame pair
        warping_err = calculate_warping_error(prev_frame, current_frame)
        clip_sim = calculate_clip_consistency(prev_frame, current_frame, clip_model, clip_preprocess, device)

        results.append({
            'frame_pair': f"{frame_num-1}-{frame_num}",
            'warping_error': warping_err,
            'clip_consistency': clip_sim
        })

        # Update the previous frame for the next iteration
        prev_frame = current_frame

        # Print progress
        sys.stdout.write(
            f"\rProcessing Frame Pair {frame_num}/{num_frames-1} | "
            f"Warping Error: {warping_err:.2f} | CLIP Consistency: {clip_sim:.4f}"
        )
        sys.stdout.flush()

    cap.release()
    print("\n" + "-" * 30)
    print("Analysis complete.")

    # --- 4. Report final average results ---
    if results:
        df = pd.DataFrame(results)
        avg_warping_error = df['warping_error'].mean()
        avg_clip_consistency = df['clip_consistency'].mean()

        print("\n--- Overall Average Temporal Consistency ---")
        print(f"Average Warping Error (Lower is better):  {avg_warping_error:.4f}")
        print(f"Average CLIP Consistency (Higher is better): {avg_clip_consistency:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the temporal consistency of a video using Warping Error and CLIP Similarity.")
    parser.add_argument("video_file", help="Path to the video file to be analyzed.")
    args = parser.parse_args()
    
    # Check if pandas is installed for reporting
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for this script.")
        print("Please install it with: pip install pandas")
        sys.exit(1)
        
    analyze_temporal_consistency(args.video_file)

'''
Required Libraries:
- OpenCV
- PyTorch
- pandas
- CLIP (from OpenAI's repository)
- PIL (Pillow)

Usage: python3 Temporal_consistency.py ./path/to/video.mp4
'''