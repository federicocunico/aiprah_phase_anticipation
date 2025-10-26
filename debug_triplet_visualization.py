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
    Overlay triplet information in a compact horizontal bar format.
    Optimized layout with minimal frame obstruction.
    
    Args:
        frame: BGR image as numpy array
        left_triplet: Left arm triplet string (GT)
        right_triplet: Right arm triplet string (GT)
        frame_idx: Frame index number
        left_pred_triplet: Left arm predicted triplet (optional)
        right_pred_triplet: Right arm predicted triplet (optional)

    Returns:
        Frame with compact overlaid triplet information
    """
    overlay_frame = frame.copy()
    height, width = overlay_frame.shape[:2]

    # Colors (BGR format)
    gt_color = (50, 255, 50)        # Bright green for GT
    pred_color = (50, 150, 255)     # Orange for predictions
    left_bg = (40, 40, 40)          # Dark gray for left arm
    right_bg = (60, 40, 40)         # Slightly different for right arm
    header_bg = (20, 20, 20)        # Very dark for header
    text_white = (255, 255, 255)
    text_gray = (180, 180, 180)
    divider_color = (100, 100, 100)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    
    # Parse triplets
    left_verb, _, left_target = parse_triplet(left_triplet)
    right_verb, _, right_target = parse_triplet(right_triplet)
    
    # Prepare prediction data if available
    left_pred_verb, left_pred_target = None, None
    right_pred_verb, right_pred_target = None, None
    if left_pred_triplet:
        left_pred_verb, _, left_pred_target = parse_triplet(left_pred_triplet)
    if right_pred_triplet:
        right_pred_verb, _, right_pred_target = parse_triplet(right_pred_triplet)
    
    # ===== TOP BAR: Compact info bar =====
    bar_height = 75  # Compact height for info bar
    
    # Create semi-transparent overlay for top bar
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), header_bg, -1)
    cv2.addWeighted(overlay, 0.85, overlay_frame, 0.15, 0, overlay_frame)
    
    # Draw divider line
    cv2.line(overlay_frame, (0, bar_height), (width, bar_height), divider_color, 2)
    
    # === Layout: Frame Index (center top) ===
    frame_text = f"Frame: {frame_idx}"
    (fw, fh), _ = cv2.getTextSize(frame_text, font_bold, 0.6, 2)
    cv2.putText(overlay_frame, frame_text, (width//2 - fw//2, 25), 
                font_bold, 0.6, text_white, 2)
    
    # === Vertical divider in the middle ===
    mid_x = width // 2
    cv2.line(overlay_frame, (mid_x, 35), (mid_x, bar_height - 5), divider_color, 1)
    
    # ===== LEFT HALF: Left Arm =====
    left_section_x = 10
    y = 50
    
    # Label
    cv2.putText(overlay_frame, "L-ARM", (left_section_x, y), 
                font, 0.45, text_gray, 1)
    
    # GT info (compact horizontal layout)
    gt_text = f"GT: {left_verb} -> {left_target}"
    cv2.putText(overlay_frame, gt_text, (left_section_x + 55, y), 
                font, 0.4, gt_color, 1)
    
    # Prediction (if available)
    y += 20
    if left_pred_verb and left_pred_target:
        pred_text = f"PR: {left_pred_verb} -> {left_pred_target}"
        cv2.putText(overlay_frame, pred_text, (left_section_x + 55, y), 
                    font, 0.4, pred_color, 1)
    else:
        cv2.putText(overlay_frame, "PR: ----", (left_section_x + 55, y), 
                    font, 0.4, text_gray, 1)
    
    # ===== RIGHT HALF: Right Arm =====
    right_section_x = mid_x + 10
    y = 50
    
    # Label
    cv2.putText(overlay_frame, "R-ARM", (right_section_x, y), 
                font, 0.45, text_gray, 1)
    
    # GT info (compact horizontal layout)
    gt_text = f"GT: {right_verb} -> {right_target}"
    cv2.putText(overlay_frame, gt_text, (right_section_x + 55, y), 
                font, 0.4, gt_color, 1)
    
    # Prediction (if available)
    y += 20
    if right_pred_verb and right_pred_target:
        pred_text = f"PR: {right_pred_verb} -> {right_pred_target}"
        cv2.putText(overlay_frame, pred_text, (right_section_x + 55, y), 
                    font, 0.4, pred_color, 1)
    else:
        cv2.putText(overlay_frame, "PR: ----", (right_section_x + 55, y), 
                    font, 0.4, text_gray, 1)
    
    # ===== BOTTOM BAR: Status indicators (minimal) =====
    bottom_bar_height = 25
    bottom_y = height - bottom_bar_height
    
    # Semi-transparent bottom bar
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (0, bottom_y), (width, height), header_bg, -1)
    cv2.addWeighted(overlay, 0.75, overlay_frame, 0.25, 0, overlay_frame)
    
    # Divider line
    cv2.line(overlay_frame, (0, bottom_y), (width, bottom_y), divider_color, 1)
    
    # Compact status: L | R
    status_y = height - 8
    status_left = f"L: {left_verb}({left_target})"
    status_right = f"R: {right_verb}({right_target})"
    
    # Truncate if needed
    max_len = 40
    if len(status_left) > max_len:
        status_left = status_left[:max_len-3] + "..."
    if len(status_right) > max_len:
        status_right = status_right[:max_len-3] + "..."
    
    cv2.putText(overlay_frame, status_left, (10, status_y), 
                font, 0.35, gt_color, 1)
    
    (sw, _), _ = cv2.getTextSize(status_right, font, 0.35, 1)
    cv2.putText(overlay_frame, status_right, (width - sw - 10, status_y), 
                font, 0.35, gt_color, 1)

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
