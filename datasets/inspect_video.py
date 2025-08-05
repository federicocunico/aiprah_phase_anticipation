import os
import cv2

target_video = "data/heichole/Videos/Full/Hei-Chole14.mp4"

# inspect video, get fps and frame count, using ffmpeg and then cv2
cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 {target_video}"
os.system(cmd)

cap = cv2.VideoCapture(target_video)
if not cap.isOpened():
    print("Error: Could not open video.")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Frame Count: {frame_count}")

