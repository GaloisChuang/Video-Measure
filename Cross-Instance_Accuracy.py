import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os
import sys
import argparse
import pandas as pd
from typing import List

# This script computes the Cross-Instance Accuracy (CIA) score for a video.
# CIA measures how well different object instances in a video match their respective text descriptions.


def get_instance_crops(frame: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract cropped regions (instances) from a single video frame based on binary masks.

    Args:
        frame (np.ndarray): The OpenCV frame in BGR format.
        masks (List[np.ndarray]): A list of binary (0 or 255) masks, one per instance.

    Returns:
        List[np.ndarray]: A list of cropped instance images (in BGR format).
    """
    crops = []
    for mask in masks:
        # Find the bounding box of the mask
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            continue  # Skip empty masks

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Crop the image using the bounding box
        cropped_instance = frame[y_min:y_max + 1, x_min:x_max + 1]
        crops.append(cropped_instance)
    return crops


def calculate_cia_score(
    video_path: str,
    mask_dir: str,
    captions: List[str],
    clip_model,
    clip_preprocess,
    device: str
) -> float:
    """
    Calculate the Cross-Instance Accuracy (CIA) score for a given video.

    Args:
        video_path (str): Path to the edited/test video file.
        mask_dir (str): Directory containing subfolders for instance masks.
                        Expected structure:
                        mask_dir/instance_000/frame_000.png, etc.
        captions (List[str]): List of text descriptions corresponding to each instance.
        clip_model: The loaded CLIP model.
        clip_preprocess: The CLIP preprocessing function.
        device (str): Device to run the computation ('cuda' or 'cpu').

    Returns:
        float: The final CIA score, between 0 and 1.
    """
    num_instances = len(captions)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 1. Extract CLIP features for each instance across all frames ---
    instance_features_per_frame = [[] for _ in range(num_instances)]

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Load masks for all instances in the current frame
        current_frame_masks = []
        for i in range(num_instances):
            mask_path = os.path.join(mask_dir, f'instance_{i:03d}', f'frame_{frame_idx:03d}.png')
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found {mask_path}, skipping this frame.")
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            current_frame_masks.append(mask)

        if len(current_frame_masks) != num_instances:
            continue

        # Crop instances
        crops = get_instance_crops(frame, current_frame_masks)

        # Compute CLIP image features for each cropped instance
        with torch.no_grad():
            for i, crop in enumerate(crops):
                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
                features = clip_model.encode_image(image_input)
                instance_features_per_frame[i].append(features)

    cap.release()

    # --- 2. Average the features across all frames for each instance ---
    avg_instance_features = []
    for features_list in instance_features_per_frame:
        if not features_list:
            print("Warning: Some instances have no valid features across all frames.")
            avg_instance_features.append(torch.zeros((1, 512), device=device))
            continue
        avg_feature = torch.stack(features_list).mean(dim=0)
        avg_instance_features.append(avg_feature)

    if not avg_instance_features:
        print("Error: Could not extract any valid instance features from the video.")
        return 0.0

    # --- 3. Compute text features ---
    with torch.no_grad():
        text_tokens = clip.tokenize(captions).to(device)
        text_features = clip_model.encode_text(text_tokens)

    # --- 4. Compute similarity matrix ---
    image_features_stack = torch.cat(avg_instance_features)

    # Normalize for cosine similarity
    image_features_stack /= image_features_stack.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity_matrix = (image_features_stack @ text_features.T).cpu().numpy()

    print("\n--- Similarity Matrix ---")
    print("Rows represent image instances; columns represent text descriptions.")
    print(pd.DataFrame(similarity_matrix,
                       columns=captions,
                       index=[f"Instance_{i}" for i in range(num_instances)]))

    # --- 5. Compute CIA score ---
    correct_matches = 0
    for i in range(num_instances):
        best_match_idx = np.argmax(similarity_matrix[i, :])
        if best_match_idx == i:
            correct_matches += 1

    cia_score = correct_matches / num_instances if num_instances > 0 else 0.0
    return cia_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Cross-Instance Accuracy (CIA) score for a video.")
    parser.add_argument("video_path", type=str, help="Path to the edited/test video file.")
    parser.add_argument("mask_dir", type=str, help="Path to the directory containing instance mask subfolders.")
    parser.add_argument("captions", type=str, nargs='+', help="Target text descriptions in order, separated by spaces.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except Exception as e:
        print(f"Error: Failed to load CLIP or dependencies.\n{e}")
        print("Install missing packages:\n"
              "pip install git+https://github.com/openai/CLIP.git\n"
              "pip install torch opencv-python-headless pandas")
        sys.exit(1)

    final_score = calculate_cia_score(args.video_path, args.mask_dir, args.captions, model, preprocess, device)

    print("\n" + "=" * 30)
    print(f"Final Cross-Instance Accuracy (CIA) Score: {final_score:.4f}")
    print("=" * 30)
