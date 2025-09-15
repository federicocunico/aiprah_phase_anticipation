#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node: online anticipation inference from an image stream + post-processing.

After each inference step we:
  - smooth logits & anticipation across last N steps,
  - enforce monotonic (non-decreasing) phase,
  - compute TTNP from the *next* phase's anticipation; if phase==last -> 0.
  - (NEW) publish completion/advancement percentage for the current phase.

Publishes both RAW and PROCESSED outputs.
"""

# =========================
# Imports
# =========================
import copy
import os

if "ROS_MASTER_URI" not in os.environ or "localhost" in os.environ["ROS_MASTER_URI"]:
    print("Setting default ROS MASTER to James")
    os.environ["ROS_MASTER_URI"] = "http://James:11311"
from threading import Lock, Thread
import time
from collections import deque
from typing import Deque, Optional, Tuple, List, Union

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image as RosImage
from std_msgs.msg import Int32, Float32, Float32MultiArray

# Import your model definition only
from anticipation_model_slow_fast import (
    SlowFastTemporalAnticipation as AnticipationTemporalModel,
)


# =========================
# RUNTIME CONFIG CONSTANTS
# =========================
IMAGE_TOPIC = "/endoscope/left/image_color/compressed"  # incoming image topic
TARGET_FPS_OUTPUT = 1.0  # Hz: throttle processing to this FPS
MAX_FPS_INPUT = 1.0
MODEL_PATH = "/home/saras/saras/linux/src/aiprah/phase_anticpation/src/v1/peg_and_ring_slowfast_transformer_BEST.pth"  # checkpoint to load
DEVICE_PREF = "cuda"  # "cuda" or "cpu"

# ROS topics to publish
ANTICIPATION_REGRESSION_TOPIC = "/anticipation/regression"  # Float32MultiArray (len C)
PHASE_PREDICTION_TOPIC = "/anticipation/phase"  # Int32
PHASE_PROBABILITY_TOPIC = "/anticipation/confidence"  # Float32MultiArray (len C)
PHASE_COMPLETION_TOPIC = "/anticipation/completion"  # Float32

QUEUE_SIZE_SUB = 10
QUEUE_SIZE_PUB = 1


# =========================
# Torchvision preprocessor
# =========================
class Preprocessing:
    def __init__(self):
        self.tf = T.Compose(
            [
                T.Resize((250, 250)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.26728517, 0.32384375, 0.2749076),
                    std=(0.15634732, 0.167153, 0.15354523),
                ),
            ]
        )

    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        if not isinstance(bgr, Image.Image):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
        else:
            pil = bgr
        return self.tf(pil)  #.contiguous()  # [3,224,224]

import torch
import torch.nn.functional as F

class BatchFastPreprocessing:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.mean = torch.tensor(
            [0.26728517, 0.32384375, 0.2749076],
            device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.15634732, 0.167153, 0.15354523],
            device=self.device
        ).view(1, 3, 1, 1)

    def __call__(self, bgr_batch: np.ndarray) -> torch.Tensor:
        # bgr_batch: (T, H, W, 3), uint8
        x = torch.from_numpy(bgr_batch)                              \
                 .permute(0, 3, 1, 2)                                \
                 .to(self.device, non_blocking=True)                \
                 .float() / 255.0                                   # (T,3,H,W), [0–1]
        x = x[:, [2, 1, 0], ...]                                     # BGR→RGB
        x = F.interpolate(
            x,
            size=(250, 250),
            mode="bilinear",
            align_corners=False
        )                                                            # resize
        crop = (250 - 224) // 2
        x = x[:, :, crop : crop + 224, crop : crop + 224]            # center‐crop
        x = (x - self.mean) / self.std                               # normalize
        return x                                                     # (T,3,224,224)


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
    SEQ_LEN = 16
    NUM_CLASSES = 6
    TIME_HORIZON = 2.0
    FUTURE_F = 1

    model = AnticipationTemporalModel(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone_fast="resnet50",
        backbone_slow="resnet50",
        pretrained_backbone_fast=True,
        pretrained_backbone_slow=True,
        freeze_backbone_fast=False,
        freeze_backbone_slow=False,
        hidden_channels=384,
        num_temporal_layers=6,
        dropout=0.1,
        use_spatial_attention=True,
        attn_heads=8,
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Bootstrap (init lazy buffers like BN, etc.)
    _inference_mode = getattr(torch, "inference_mode", None)
    ctx = _inference_mode() if _inference_mode is not None else torch.no_grad()
    with ctx:
        dummy_frames = torch.randn(1, SEQ_LEN, 3, 224, 224, device=device)
        dummy_prevA = torch.zeros(1, FUTURE_F, NUM_CLASSES, device=device)
        _ = model(dummy_frames, dummy_prevA)

    return model


# =========================
# Node
# =========================
class AnticipationNode(Thread):
    def __init__(self):
        super().__init__()

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
        rospy.loginfo(
            f"[anticipation_node] Model specs: T={self.T}, M={self.M}, C={self.C}, H={self.H:g}"
        )

        # preprocessing + buffers
        self.preproc  = BatchFastPreprocessing()
        self.frame_buf: Deque[torch.Tensor] = deque(maxlen=self.T)
        self.prevA_buf: Deque[torch.Tensor] = deque(maxlen=self.M)

        # throttling
        self.min_period_out = 1.0 / max(TARGET_FPS_OUTPUT, 1e-6)
        self.last_proc_time: Optional[float] = None

        self.min_period_in = 1.0 / max(MAX_FPS_INPUT, 1e-6)

        self.start_time = time.time()

        self._lock = Lock()
        self.data = None

        # ROS I/O
        self.pub_phase = rospy.Publisher(
            PHASE_PREDICTION_TOPIC, Int32, queue_size=QUEUE_SIZE_PUB
        )
        self.pub_anticip = rospy.Publisher(
            ANTICIPATION_REGRESSION_TOPIC, Float32MultiArray, queue_size=QUEUE_SIZE_PUB
        )

        self.pub_phase_prob = rospy.Publisher(
            PHASE_PROBABILITY_TOPIC, Float32, queue_size=QUEUE_SIZE_PUB
        )

        # NEW: completion/advancement publisher
        self.pub_completion = rospy.Publisher(
            PHASE_COMPLETION_TOPIC, Float32, queue_size=QUEUE_SIZE_PUB
        )

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(
            IMAGE_TOPIC, CompressedImage, self.image_cb, queue_size=QUEUE_SIZE_SUB
        )

        rospy.loginfo("[anticipation_node] Ready. Waiting for images...")
        self._reading_started = False
        

    def run(self):
        prev_exec = time.time()
        while True:
            if self.data is None:
                time.sleep(0.1)
                continue
            
            with self._lock:
                if self.data is not None:
                    data = copy.deepcopy(self.data)
                self.data = None
            
            now = time.time()
            elapsed =  now - prev_exec
            if elapsed < self.min_period_out:
                continue
            # rospy.loginfo(f"Writing at {1/elapsed}")
            prev_exec = now
            
            self.pub_anticip.publish(Float32MultiArray(data=data["regression"]))
            self.pub_phase.publish(Int32(data=data["phase"]))
            self.pub_phase_prob.publish(Float32(data=data["conf"]))
            self.pub_completion.publish(Float32(data=data["completion"]))


    # ---------- helpers ----------
    def _set_placeholder(self):
        pred_arr = [0.0] + [self.H] * (self.C - 1)
        # self.pub_anticip.publish(Float32MultiArray(data=pred_arr))
        # self.pub_phase.publish(Int32(data=0))
        # self.pub_phase_prob.publish(Float32(data=-1.0))  # not a valid confidence because no inference
        # self.pub_completion.publish(Float32(data=0.0))
        with self._lock:
            self.data = {
                "regression": pred_arr,
                "phase": 0,
                "conf": -1.0,
                "completion": 0
            }

    @staticmethod
    def _extract_completion_for_phase(
        completion_tensor: torch.Tensor, current_phase: int, num_classes: int
    ) -> float:
        """
        Turns the model's completion output into a scalar to publish.

        - If scalar -> publish item().
        - If vector with len==num_classes -> publish value for current_phase.
        - Otherwise -> publish mean() as a fallback.
        """
        try:
            t = completion_tensor.detach().cpu().float()
        except Exception:
            # If it's already a float
            try:
                return float(completion_tensor)
            except Exception:
                return 0.0

        if t.numel() == 1:
            return float(t.item())
        if t.dim() > 0 and t.numel() == num_classes:
            # Keep indexing robust to potential extra dims
            v = t.view(-1)
            idx = int(np.clip(current_phase, 0, num_classes - 1))
            return float(v[idx].item())
        # Fallback: mean
        return float(t.mean().item())

    # ---------- callback ----------
    def image_cb(self, msg: Union[CompressedImage, RosImage]):
        if not self._reading_started:
            self._reading_started = True
            rospy.loginfo("Image received; starting.")

        # Exit if reading too fast w.r.t. self.min_period_in
        # now = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        now = time.time()
        if self.last_proc_time is not None and (
           now - self.last_proc_time < self.min_period_in
        ):
            return
        self.last_proc_time = now

        try:
            if isinstance(msg, RosImage):
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            elif isinstance(msg, CompressedImage):
                image_data = np.frombuffer(msg.data, np.uint8)
                bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            # Optional debug dump:
            # cv2.imwrite("TEST.png", bgr)
        except Exception as e:
            rospy.logwarn(f"[anticipation_node] cv_bridge error: {e}")
            return

        self.frame_buf.append(bgr)

        if len(self.frame_buf) < self.T:
            rospy.loginfo(f"Image Buffer not ready, publishing placeholder {len(self.frame_buf)}/{self.T}")
            self._set_placeholder()
            return

        # Prefer inference_mode if available; otherwise no_grad.
        # _inference_mode = getattr(torch, "inference_mode", None)
        # ctx = _inference_mode() if _inference_mode is not None else torch.no_grad()

        with torch.no_grad():
            s = time.time()
            bgr_batch = np.stack(self.frame_buf, axis=0)  #
            e1 = time.time() -s 
            s = time.time()
            batch_tensor = self.preproc(bgr_batch).unsqueeze(0)
            e2 = time.time() -s
            s = time.time()
            reg, _logits_unused, completion_perc, _ = self.model(batch_tensor)  # noqa: F841
            e3 = time.time() - s
            reg = reg.squeeze(0)  # [C]

        phase_proc_int = reg.argmin().item()  # Do not use logits
        probs = F.softmax(-reg, dim=0)  # for potential future use
        confidence_scalar = float(probs[phase_proc_int].item())
        reg_raw_list = reg.detach().cpu().float().tolist()
        completion_scalar = completion_perc.detach().cpu().float().item()

        with self._lock:
            self.data = {
                "regression": reg_raw_list,
                "phase": int(phase_proc_int),
                "conf": confidence_scalar,
                "completion": completion_scalar
            }

        if False:        
            s = f"Total time: {e1+e2+e3} s ({1/(e1+e2+e3)} FPS)"+\
                f"\n\tStack {e1} s"+\
                f"\n\tPreprocessing + CPU-to-GPU {e2} s"+\
                f"\n\tInference {e3} s"
            rospy.loginfo(s)



# =========================
# main
# =========================
def main():


    # # Load and prepare model
    # model = load_model("cuda")
    # model.eval()

    # # Warm-up
    # for _ in range(10):
    #     x = torch.randn(1, 16, 3, 224, 224, device="cuda")
    #     with torch.no_grad():
    #         _ = model(x)

    # # Benchmark loop
    # n_iters = 100
    # torch.cuda.synchronize()
    # start = time.perf_counter()
    # for _ in range(n_iters):
    #     x = torch.randn(1, 16, 3, 224, 224, device="cuda")
    #     with torch.no_grad():
    #         _ = model(x)
    # torch.cuda.synchronize()
    # end = time.perf_counter()

    # total_time = end - start
    # fps = n_iters / total_time

    # print(f"Average latency: {total_time/n_iters*1000:.2f} ms")
    # print(f"Average FPS: {fps:.2f}")

    rospy.init_node("anticipation_infer_node", anonymous=False)
    node = AnticipationNode()
    node.start()
    rospy.spin()


if __name__ == "__main__":
    main()
