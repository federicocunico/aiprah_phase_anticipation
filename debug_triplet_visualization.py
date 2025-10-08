#!/usr/bin/env python3
"""
Debug visualization tool for dual-arm action triplets.
Creates a video showing each frame with overlaid triplet annotations.
OPTIMIZED for lightning-fast data loading and processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from tqdm import tqdm
from datasets.peg_and_ring_workflow import PegAndRing
from PIL import Image, ImageFont, ImageDraw
import tempfile
import os
import pickle
import json


# Global cache for triplet data - load once, use many times
_TRIPLET_CACHE = None
_FRAME_CACHE = None


def _load_triplet_cache_fast():
    """Load triplet data super fast with caching."""
    global _TRIPLET_CACHE, _FRAME_CACHE

    cache_file = Path("debug_outputs/triplet_cache.pkl")

    # Try to load from cache first
    if cache_file.exists() and _TRIPLET_CACHE is None:
        try:
            print("Loading triplet cache...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                _TRIPLET_CACHE = cache_data["triplets"]
                _FRAME_CACHE = cache_data["frames"]
            print(f"✅ Loaded cached data for {len(_TRIPLET_CACHE)} videos")
            return
        except:
            print("❌ Cache corrupted, rebuilding...")

    # Build cache if not exists
    if _TRIPLET_CACHE is None:
        print("🔄 Building triplet cache (one-time setup)...")

        # Load dataset once
        ds_train = PegAndRing(
            root_dir="data/peg_and_ring_workflow",
            mode="train",
            seq_len=1,
            stride=1,
            fps=1,
            time_unit="minutes",
            force_triplets=True,
        )

        _TRIPLET_CACHE = {}
        _FRAME_CACHE = {}

        # Process all data in one pass
        for i in tqdm(range(len(ds_train)), desc="Caching triplet data"):
            frames, meta = ds_train[i]
            video_name = meta["video_name"]

            if video_name not in _TRIPLET_CACHE:
                _TRIPLET_CACHE[video_name] = []
                _FRAME_CACHE[video_name] = []

            frame_data = {
                "frame_idx": meta["frames_indexes"][0].item(),
                "left_verb": meta["triplet_left_verb"],
                "left_destination": meta["triplet_left_destination"],
                "right_verb": meta["triplet_right_verb"],
                "right_destination": meta["triplet_right_destination"],
                "frame_path": meta["frames_filepath"][0],
            }

            _TRIPLET_CACHE[video_name].append(frame_data)

        # Sort by frame index for each video
        for video_name in _TRIPLET_CACHE:
            _TRIPLET_CACHE[video_name].sort(key=lambda x: x["frame_idx"])

        # Save cache
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"triplets": _TRIPLET_CACHE, "frames": _FRAME_CACHE}, f)

        print(f"💾 Cache saved with {len(_TRIPLET_CACHE)} videos")


def debug_triplet(
    video_index: int, output_dir: str = "debug_outputs", fps_out: int = 1
):
    """Lightning-fast triplet debug video creation."""

    # Load cache super fast
    _load_triplet_cache_fast()

    video_name = f"video{video_index:02d}"

    if video_name not in _TRIPLET_CACHE:
        available = list(_TRIPLET_CACHE.keys())
        print(f"❌ Video {video_name} not found. Available: {available}")
        return

    print(
        f"Creating debug video for {video_name} with {len(_TRIPLET_CACHE[video_name])} frames..."
    )

    out_video = Path(output_dir) / f"video{video_index:02d}_triplet_debug.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Use original video file for maximum speed
    original_video_path = Path("data/peg_and_ring_workflow") / f"{video_name}.mp4"

    if not original_video_path.exists():
        print(f"❌ Original video not found: {original_video_path}")
        return

    # Open original video
    cap = cv2.VideoCapture(str(original_video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {original_video_path}")
        return

    # Process frames super fast
    frame_data_list = _TRIPLET_CACHE[video_name]

    # Get video properties
    first_frame = cv2.imread(frame_data_list[0]["frame_path"])
    original_height, original_width = first_frame.shape[:2]

    # Use original frame dimensions (no more expanded layout)
    layout_width = original_width
    layout_height = original_height

    # Setup video writer with original frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(out_video), fourcc, fps_out, (layout_width, layout_height)
    )

    for frame_data in tqdm(frame_data_list, desc=f"Processing {video_name}"):
        frame_idx = frame_data["frame_idx"]
        frame = cv2.imread(frame_data["frame_path"])
        if frame is None:
            print(f"❌ Failed to read frame: {frame_data['frame_path']}")
            continue

        # Build triplet strings
        left_triplet = (
            f"{frame_data['left_verb']}(left_arm,{frame_data['left_destination']})"
        )
        right_triplet = (
            f"{frame_data['right_verb']}(right_arm,{frame_data['right_destination']})"
        )

        # Add overlays
        annotated_frame = add_triplet_overlays(
            frame, left_triplet, right_triplet, frame_idx
        )
        video_writer.write(annotated_frame)

    # Cleanup
    video_writer.release()

    print(f"Triplet debug video saved to {out_video}")


def parse_triplet(triplet_str: str) -> Tuple[str, str, str]:
    """Parse triplet string like 'reach(left_arm,green_peg)' into components."""
    import re

    pattern = r"(\w+)\(([^,]+),([^)]+)\)"
    match = re.match(pattern, triplet_str)
    if match:
        verb = match.group(1)
        subject = match.group(2)
        target = match.group(3)
        return verb, subject, target
    return "null-verb", "null-subject", "null-target"


def add_triplet_overlays(
    frame: np.ndarray,
    left_triplet: str,
    right_triplet: str,
    frame_idx: int,
    left_pred_triplet: str = None,
    right_pred_triplet: str = None,
) -> np.ndarray:
    """
    Overlay triplet information directly on the frame.
    
    Left side: Left arm information (GT + predictions)
    Right side: Right arm information (GT + predictions)
    
    Args:
        frame: BGR image as numpy array
        left_triplet: Left arm triplet string (GT)
        right_triplet: Right arm triplet string (GT)
        frame_idx: Frame index number
        left_pred_triplet: Left arm predicted triplet (optional)
        right_pred_triplet: Right arm predicted triplet (optional)

    Returns:
        Frame with overlaid triplet information
    """
    # Work on a copy of the frame
    overlay_frame = frame.copy()
    height, width = overlay_frame.shape[:2]

    # Colors
    gt_color = (0, 255, 0)  # Green for ground truth
    pred_color = (0, 100, 255)  # Orange for predictions
    text_color = (255, 255, 255)  # White for labels
    dash_color = (100, 100, 100)  # Gray for dashes
    bg_color = (0, 0, 0)  # Black background for text

    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    small_font_scale = 0.4
    thickness = 2
    line_height = 20

    # Helper function to add text with background
    def add_text_with_bg(img, text, pos, color, bg_color=bg_color, font_scale=font_scale, thickness=thickness):
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(img, 
                      (pos[0] - 2, pos[1] - text_height - 2), 
                      (pos[0] + text_width + 2, pos[1] + baseline + 2), 
                      bg_color, -1)
        
        # Draw text
        cv2.putText(img, text, pos, font, font_scale, color, thickness)

    # Parse triplets
    left_verb, left_subject, left_target = parse_triplet(left_triplet)
    right_verb, right_subject, right_target = parse_triplet(right_triplet)

    # === LEFT SIDE (Left arm information) ===
    left_x = 10
    y_pos = 30

    # Frame index at top
    add_text_with_bg(overlay_frame, f"Frame: {frame_idx}", (width//2 - 50, y_pos), text_color, font_scale=0.6)
    
    # Left arm section
    y_pos = 60
    add_text_with_bg(overlay_frame, "LEFT ARM", (left_x, y_pos), text_color, font_scale=0.6)
    y_pos += line_height + 5

    # Ground truth
    add_text_with_bg(overlay_frame, "GT:", (left_x, y_pos), text_color, font_scale=small_font_scale)
    y_pos += line_height
    add_text_with_bg(overlay_frame, f"  {left_verb}", (left_x, y_pos), gt_color, font_scale=small_font_scale)
    y_pos += line_height
    add_text_with_bg(overlay_frame, f"  {left_target}", (left_x, y_pos), gt_color, font_scale=small_font_scale)
    y_pos += line_height + 5

    # Predictions (if available)
    if left_pred_triplet:
        pred_verb, pred_subject, pred_target = parse_triplet(left_pred_triplet)
        add_text_with_bg(overlay_frame, "PRED:", (left_x, y_pos), text_color, font_scale=small_font_scale)
        y_pos += line_height
        add_text_with_bg(overlay_frame, f"  {pred_verb}", (left_x, y_pos), pred_color, font_scale=small_font_scale)
        y_pos += line_height
        add_text_with_bg(overlay_frame, f"  {pred_target}", (left_x, y_pos), pred_color, font_scale=small_font_scale)
    else:
        add_text_with_bg(overlay_frame, "PRED: ----", (left_x, y_pos), dash_color, font_scale=small_font_scale)

    # === RIGHT SIDE (Right arm information) ===
    right_x = width - 150  # Position from right edge
    y_pos = 60

    # Right arm section
    add_text_with_bg(overlay_frame, "RIGHT ARM", (right_x, y_pos), text_color, font_scale=0.6)
    y_pos += line_height + 5

    # Ground truth
    add_text_with_bg(overlay_frame, "GT:", (right_x, y_pos), text_color, font_scale=small_font_scale)
    y_pos += line_height
    add_text_with_bg(overlay_frame, f"  {right_verb}", (right_x, y_pos), gt_color, font_scale=small_font_scale)
    y_pos += line_height
    add_text_with_bg(overlay_frame, f"  {right_target}", (right_x, y_pos), gt_color, font_scale=small_font_scale)
    y_pos += line_height + 5

    # Predictions (if available)
    if right_pred_triplet:
        pred_verb, pred_subject, pred_target = parse_triplet(right_pred_triplet)
        add_text_with_bg(overlay_frame, "PRED:", (right_x, y_pos), text_color, font_scale=small_font_scale)
        y_pos += line_height
        add_text_with_bg(overlay_frame, f"  {pred_verb}", (right_x, y_pos), pred_color, font_scale=small_font_scale)
        y_pos += line_height
        add_text_with_bg(overlay_frame, f"  {pred_target}", (right_x, y_pos), pred_color, font_scale=small_font_scale)
    else:
        add_text_with_bg(overlay_frame, "PRED: ----", (right_x, y_pos), dash_color, font_scale=small_font_scale)

    # Add bottom status line with full triplets (smaller text)
    bottom_y = height - 30
    full_left = f"L: {left_triplet}"
    full_right = f"R: {right_triplet}"
    
    # Truncate if too long
    max_chars = 35
    if len(full_left) > max_chars:
        full_left = full_left[:max_chars-3] + "..."
    if len(full_right) > max_chars:
        full_right = full_right[:max_chars-3] + "..."
    
    add_text_with_bg(overlay_frame, full_left, (10, bottom_y), gt_color, font_scale=0.4)
    
    # Calculate position for right text to align to right side
    (text_width, _), _ = cv2.getTextSize(full_right, font, 0.4, 1)
    right_text_x = width - text_width - 10
    add_text_with_bg(overlay_frame, full_right, (right_text_x, bottom_y), gt_color, font_scale=0.4)

    return overlay_frame


def list_available_videos():
    """Lightning-fast video listing using cache."""
    _load_triplet_cache_fast()

    video_indices = []
    for video_name in _TRIPLET_CACHE.keys():
        try:
            video_num = int(video_name.replace("video", ""))
            video_indices.append(video_num)
        except ValueError:
            continue

    video_indices = sorted(video_indices)
    print(f"Available videos: {video_indices} ({len(video_indices)} total)")

    return video_indices


def get_video_info(video_index: int):
    """Get quick info about a video without loading dataset."""
    _load_triplet_cache_fast()

    video_name = f"video{video_index:02d}"
    if video_name not in _TRIPLET_CACHE:
        print(f"❌ Video {video_name} not found")
        return None

    frames = _TRIPLET_CACHE[video_name]
    unique_left_verbs = set(f["left_verb"] for f in frames)
    unique_right_verbs = set(f["right_verb"] for f in frames)
    unique_left_targets = set(f["left_destination"] for f in frames)
    unique_right_targets = set(f["right_destination"] for f in frames)

    print(f"📊 {video_name} Info:")
    print(f"  Frames: {len(frames)}")
    print(f"  Left verbs: {unique_left_verbs}")
    print(f"  Right verbs: {unique_right_verbs}")
    print(f"  Left targets: {unique_left_targets}")
    print(f"  Right targets: {unique_right_targets}")

    return len(frames)


def main_all():
    """Create debug videos for all available videos - FAST!"""
    video_list = list_available_videos()
    print(f"🚀 Creating debug videos for {len(video_list)} videos...")

    for vid in tqdm(video_list, desc="Creating debug videos"):
        debug_triplet(vid)
    
    # zip all outputs in a final zip
    cmd = "cd debug_outputs && zip -r triplet_debug_videos.zip ./*.mp4"
    os.system(cmd)


def clear_cache():
    """Clear the triplet cache to force rebuild."""
    global _TRIPLET_CACHE, _FRAME_CACHE
    _TRIPLET_CACHE = None
    _FRAME_CACHE = None

    cache_file = Path("debug_outputs/triplet_cache.pkl")
    if cache_file.exists():
        cache_file.unlink()
        print("🗑️ Cache cleared")


if __name__ == "__main__":
    main_all()
