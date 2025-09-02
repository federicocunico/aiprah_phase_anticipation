#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node: online anticipation inference from an image stream + post-processing.

After each inference step we:
  - smooth logits & anticipation across last N steps,
  - enforce monotonic (non-decreasing) phase,
  - compute TTNP from the *next* phase's anticipation; if phase==last -> 0.

Publishes both RAW and PROCESSED outputs.
"""

# =========================
# Imports
# =========================
import os
if "ROS_MASTER_URI" not in os.environ or "localhost" in os.environ["ROS_MASTER_URI"]:
    print("Setting default ROS MASTER to James")
    os.environ["ROS_MASTER_URI"] = "http://James:11311"
import time
from collections import deque
from typing import Deque, Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T

import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Int32, Float32, Float32MultiArray

# Import your model definition only
from anticipation_model import TemporalCNNAnticipation as AnticipationTemporalModel


# =========================
# RUNTIME CONFIG CONSTANTS
# =========================
IMAGE_TOPIC         = "/endoscope/left/image_color/compressed"        # incoming image topic
TARGET_FPS          = 5.0                               # Hz: throttle processing to this FPS
MODEL_PATH          = "/home/saras/saras/linux/src/aiprah/phase_anticpation/src/peg_and_ring_cnn_clamp.pth"  # checkpoint to load
DEVICE_PREF         = "cuda"                            # "cuda" or "cpu"

# Smoothing / reasoning
SMOOTH_WINDOW_N     = 10                                # (legacy deque not used by new policy; kept for compatibility)
PUBLISH_RAW         = True                              # also publish raw logits/reg

# ROS topics to publish
ANTICIPATION_TOPIC_RAW   = "/anticipation/anticip_raw"         # Float32MultiArray (len C)
PHASE_TOPIC_RAW          = "/anticipation/phase_raw"           # Int32
ANTICIPATION_TOPIC_SMOOTH= "/anticipation/anticip_smooth"      # Float32MultiArray (len C)
PHASE_TOPIC_SMOOTH       = "/anticipation/phase_smooth"        # Int32
TTNP_TOPIC               = "/anticipation/ttnp"                # Float32 (scalar: time to next phase)

QUEUE_SIZE_SUB      = 10
QUEUE_SIZE_PUB      = 10


# =========================
# Torchvision preprocessor
# =========================
class Preprocessing:
    def __init__(self):
        self.tf = T.Compose([
            T.Resize((250, 250)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.26728517, 0.32384375, 0.2749076),
                        std=(0.15634732, 0.167153, 0.15354523)),
        ])

    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.tf(pil).contiguous()   # [3,224,224]


# =========================
# High-level policy (constraint logic)
# =========================
class HighLevelPolicy:
    """
    Rule-based constraint layer on top of model outputs.

    - Phase is chosen *only* from reg: phase = argmin(smoothed_reg)
    - Monotonic: can't go back to earlier phases.
    - Must observe phase 0 once before allowing >0.
    - Smooths reg over time via a time-aware EMA (inertia).
      alpha = 1 - exp(-dt / tau_seconds)
    """

    def __init__(self, num_classes: int, horizon_H: float, tau_seconds: float = 0.5, start_phase: int = 0):
        self.C = int(num_classes)
        self.H = float(horizon_H)
        self.tau = float(tau_seconds)              # smoothing time constant (s)
        self.start_phase = int(start_phase)

        self.smoothed_reg: Optional[np.ndarray] = None  # shape [C]
        self.current_phase: int = self.start_phase
        self.seen_phase0: bool = False
        self.last_seen_time: List[Optional[float]] = [None] * self.C
        self.last_time: Optional[float] = None

    def reset(self):
        self.smoothed_reg = None
        self.current_phase = self.start_phase
        self.seen_phase0 = False
        self.last_seen_time = [None] * self.C
        self.last_time = None

    def update(self, reg_tensor: torch.Tensor, now: float) -> Tuple[int, np.ndarray, float, int]:
        """
        Args:
            reg_tensor: torch.Tensor [C] (on any device)
            now: float wall/ROS time (seconds)

        Returns:
            phase_proc_int: int
            smoothed_reg_np: np.ndarray [C]
            ttnp_scalar_float: float
            phase_raw_from_reg: int  (argmin on *raw* reg)
        """
        # Convert to numpy (CPU) for policy
        reg_np = reg_tensor.detach().cpu().float().numpy()

        # Initialize smoothing state
        if self.smoothed_reg is None:
            self.smoothed_reg = reg_np.copy()
            dt = None
            alpha = 1.0  # first sample: take it
        else:
            dt = (now - self.last_time) if self.last_time is not None else None
            if dt is None or dt <= 0.0:
                alpha = 1.0
            else:
                alpha = 1.0 - np.exp(-dt / max(self.tau, 1e-6))
                alpha = float(np.clip(alpha, 0.0, 1.0))

            self.smoothed_reg = (1.0 - alpha) * self.smoothed_reg + alpha * reg_np

        # --- Phase proposals ---
        # Raw (unsmoothed) phase from reg, for RAW topic
        phase_raw_from_reg = int(np.argmin(reg_np))

        # Smoothed phase (policy uses this)
        candidate_phase = int(np.argmin(self.smoothed_reg))

        # Track "last time seen" for the candidate
        self.last_seen_time[candidate_phase] = now

        # --- Constraints ---
        if not self.seen_phase0:
            # Must start at phase 0; clamp until 0 is observed
            if candidate_phase == 0:
                self.seen_phase0 = True
                self.current_phase = 0
            else:
                self.current_phase = 0  # hold at 0 until we truly see 0
        else:
            # Monotonic non-decreasing
            if candidate_phase >= self.current_phase:
                self.current_phase = candidate_phase
            # else: keep current (can't go back)

        # --- TTNP from next phase's smoothed reg ---
        if self.current_phase >= self.C - 1:
            ttnp_scalar_float = 0.0
        else:
            next_idx = self.current_phase + 1
            ttnp_scalar_float = float(self.smoothed_reg[next_idx])

        self.last_time = now

        return self.current_phase, self.smoothed_reg.copy(), ttnp_scalar_float, phase_raw_from_reg

    def get_last_seen_times(self) -> List[Optional[float]]:
        return list(self.last_seen_time)


# =========================
# Model Loader
# =========================
def load_model(device: torch.device) -> AnticipationTemporalModel:
    """
    Initialize AnticipationTemporalModel with the same defaults as training,
    load weights, set eval, bootstrap once, and return it.
    Only AnticipationTemporalModel is imported from anticipation_model.
    """
    # Defaults from your training script
    SEQ_LEN          = 16
    NUM_CLASSES      = 6
    TIME_HORIZON     = 2.0
    FUTURE_F         = 1

    model = AnticipationTemporalModel(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet18",              # Start light
        pretrained_backbone=True,
        freeze_backbone=False,
        hidden_channels=256,              # Main hidden dimension
        num_temporal_layers=5,            # 5 dilated conv layers
        dropout=0.1,
        use_spatial_attention=True,       # Learn important spatial regions
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Bootstrap (init lazy buffers like BN, etc.)
    _inference_mode = getattr(torch, "inference_mode", None)
    ctx = _inference_mode() if _inference_mode is not None else torch.no_grad()
    with ctx:
        dummy_frames = torch.randn(1, SEQ_LEN, 3, 224, 224, device=device)
        dummy_prevA  = torch.zeros(1, FUTURE_F, NUM_CLASSES, device=device)
        _ = model(dummy_frames, dummy_prevA)

    return model


# =========================
# Node
# =========================
class AnticipationNode:
    def __init__(self):
        # device
        if DEVICE_PREF == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        rospy.loginfo(f"[anticipation_node] Using device: {self.device}")

        # (Optional) cuDNN autotuner for fixed-size inputs.
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # load model
        self.model = load_model(self.device)

        # runtime specs (read from model)
        self.T = int(getattr(self.model, "T"))
        self.M = int(getattr(self.model, "M", 1))
        self.C = int(getattr(self.model, "C"))
        self.H = float(getattr(self.model, "H"))
        self.MAX_PHASE = self.C - 1
        rospy.loginfo(f"[anticipation_node] Model specs: T={self.T}, M={self.M}, C={self.C}, H={self.H:g}")

        # preprocessing + buffers
        self.preproc = Preprocessing()
        self.frame_buf: Deque[torch.Tensor] = deque(maxlen=self.T)
        self.prevA_buf: Deque[torch.Tensor] = deque(maxlen=self.M)

        # (legacy) smoothing history — unused by new policy
        self.hist_logits: Deque[torch.Tensor] = deque(maxlen=SMOOTH_WINDOW_N)  # each [C]
        self.hist_reg:    Deque[torch.Tensor] = deque(maxlen=SMOOTH_WINDOW_N)  # each [C]

        # throttling
        self.min_period = 1.0 / max(TARGET_FPS, 1e-6)
        self.last_proc_time: Optional[float] = None

        self.start_time = time.time()

        # High-level policy (super-logic)
        # tau_seconds controls smoothing inertia; tune as you like (e.g., 0.4–1.0s)
        self.policy = HighLevelPolicy(num_classes=self.C, horizon_H=self.H, tau_seconds=0.6, start_phase=0)

        # ROS I/O
        self.pub_phase_raw   = rospy.Publisher(PHASE_TOPIC_RAW, Int32, queue_size=QUEUE_SIZE_PUB) if PUBLISH_RAW else None
        self.pub_pred_raw    = rospy.Publisher(ANTICIPATION_TOPIC_RAW, Float32MultiArray, queue_size=QUEUE_SIZE_PUB) if PUBLISH_RAW else None

        self.pub_phase_smooth= rospy.Publisher(PHASE_TOPIC_SMOOTH, Int32, queue_size=QUEUE_SIZE_PUB)
        self.pub_pred_smooth = rospy.Publisher(ANTICIPATION_TOPIC_SMOOTH, Float32MultiArray, queue_size=QUEUE_SIZE_PUB)
        self.pub_ttnp        = rospy.Publisher(TTNP_TOPIC, Float32, queue_size=QUEUE_SIZE_PUB)

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(IMAGE_TOPIC, RosImage, self.image_cb, queue_size=QUEUE_SIZE_SUB)

        rospy.loginfo("[anticipation_node] Ready. Waiting for images...")

    # ---------- helpers ----------
    def _pad_prev_memory(self) -> torch.Tensor:
        L = len(self.prevA_buf)
        if L >= self.M:
            sel = list(self.prevA_buf)[-self.M:]
        else:
            pad = torch.tensor([0.0] + [self.H] * (self.C - 1), dtype=torch.float32)
            sel = [pad] * (self.M - L) + list(self.prevA_buf)
        out = torch.stack(sel, dim=0).unsqueeze(0)
        return out.to(self.device)

    def _publish_placeholder(self):
        pred_arr = [0.0] + [self.H] * (self.C - 1)
        if PUBLISH_RAW:
            self.pub_pred_raw.publish(Float32MultiArray(data=pred_arr))
            self.pub_phase_raw.publish(Int32(data=0))
        self.pub_pred_smooth.publish(Float32MultiArray(data=pred_arr))
        self.pub_phase_smooth.publish(Int32(data=0))
        self.pub_ttnp.publish(Float32(data=float(self.H)))

    # ---------- callback ----------
    def image_cb(self, msg: RosImage):
        now = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        if self.last_proc_time is not None and (now - self.last_proc_time < self.min_period):
            return
        self.last_proc_time = now

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite("TEST.png", bgr)
        except Exception as e:
            rospy.logwarn(f"[anticipation_node] cv_bridge error: {e}")
            return

        frame_t = self.preproc(bgr)
        self.frame_buf.append(frame_t)

        if len(self.frame_buf) < self.T:
            self._publish_placeholder()
            return

        # Prefer inference_mode if available; otherwise no_grad.
        _inference_mode = getattr(torch, "inference_mode", None)
        ctx = _inference_mode() if _inference_mode is not None else torch.no_grad()

        with ctx:
            frames = torch.stack(list(self.frame_buf), dim=0).unsqueeze(0).to(self.device)
            # prevA  = self._pad_prev_memory()
            reg, _logits_unused = self.model(frames)  # logits ignored by design
            reg = reg.squeeze(0)       # [C]

        # ---- High-level policy (from reg only) ----
        phase_proc_int, reg_smooth_np, ttnp_scalar_float, phase_raw_from_reg = self.policy.update(reg, now)

        # Prepare raw/smooth arrays
        reg_raw_list = reg.detach().cpu().float().tolist()
        reg_smooth_list = reg_smooth_np.astype(np.float32).tolist()

        # RAW publications (from *raw* reg)
        if PUBLISH_RAW:
            self.pub_pred_raw.publish(Float32MultiArray(data=reg_raw_list))
            self.pub_phase_raw.publish(Int32(data=int(phase_raw_from_reg)))

        # PROCESSED publications (policy outputs)
        self.pub_pred_smooth.publish(Float32MultiArray(data=reg_smooth_list))
        self.pub_phase_smooth.publish(Int32(data=int(phase_proc_int)))
        self.pub_ttnp.publish(Float32(data=float(ttnp_scalar_float)))

    def spin(self):
        rospy.spin()


# =========================
# main
# =========================
def main():
    rospy.init_node("anticipation_infer_node", anonymous=False)
    node = AnticipationNode()
    node.spin()


if __name__ == "__main__":
    main()