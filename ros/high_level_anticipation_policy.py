#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node: High-Level Policy microservice.

Subscribes to RAW model outputs from the inference node:
  - /anticipation/regression        (Float32MultiArray [C])  <-- REQUIRED
  - /anticipation/phase             (Int32)                  <-- OPTIONAL (for reference)
  - /anticipation/completion        (Float32)                <-- OPTIONAL (if present)

Implements the policy:
  - Smooths regression via time-aware EMA (inertia, tau_seconds).
  - Enforces monotonic (non-decreasing) phase choice from the *smoothed* regression.
  - Requires observing phase 0 once before allowing >0 (bootstrapping).
  - Computes TTNP from the next phase's **smoothed** regression (0.0 at last phase).

Publishes PROCESSED outputs:
  - /anticipation/anticip_smooth    (Float32MultiArray [C])
  - /anticipation/phase_smooth      (Int32)
  - /anticipation/ttnp              (Float32)
  - /anticipation/completion_smooth (Float32)  [optional EMA smoothing; see params]

Config (ROS params, all optional):
  ~tau_seconds: float   (default 0.6)  - EMA time constant (seconds)
  ~start_phase: int     (default 0)    - initial phase clamp until phase 0 is observed
  ~horizon_H: float     (default 2.0)  - only used for placeholder/limits
  ~smooth_completion: bool (default false) - if true, EMA-smooth completion and publish *_smooth topic
  ~completion_tau_seconds: float (default equals ~tau_seconds) - EMA for completion
  ~topics:
       regression: str (default '/anticipation/regression')
       phase:      str (default '/anticipation/phase')
       completion: str (default '/anticipation/completion')
       anticip_smooth: str (default '/anticipation/anticip_smooth')
       phase_smooth:   str (default '/anticipation/phase_smooth')
       ttnp:           str (default '/anticipation/ttnp')
       completion_smooth: str (default '/anticipation/completion_smooth')
"""

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import Int32, Float32, Float32MultiArray


# -------------------------
# High-Level Policy object
# -------------------------
class HighLevelPolicy:
    """
    Constraint layer on top of regression vector "reg" (shape [C]):

      - Phase is chosen from the *smoothed* reg: phase = argmin(smoothed_reg)
      - Monotonic: phase cannot decrease
      - Must observe phase 0 at least once before allowing > 0
      - EMA smoothing with time-aware alpha = 1 - exp(-dt / tau_seconds)
      - TTNP = smoothed_reg[current_phase + 1] (or 0 at last phase)
    """

    def __init__(self, num_classes: int, horizon_H: float, tau_seconds: float = 0.6, start_phase: int = 0):
        self.C = int(num_classes)
        self.H = float(horizon_H)
        self.tau = float(tau_seconds)
        self.start_phase = int(start_phase)

        self.smoothed_reg: Optional[np.ndarray] = None
        self.current_phase: int = self.start_phase
        self.seen_phase0: bool = False
        self.last_time: Optional[float] = None

    def reset(self):
        self.smoothed_reg = None
        self.current_phase = self.start_phase
        self.seen_phase0 = False
        self.last_time = None

    def update(self, reg_np: np.ndarray, now: float) -> Tuple[int, np.ndarray, float]:
        """
        Args:
          reg_np: numpy array [C], raw regression from model (CPU)
          now:    float seconds (wall/ROS time)

        Returns:
          phase_proc_int: int
          smoothed_reg:   np.ndarray [C]
          ttnp:           float
        """
        assert reg_np.ndim == 1, "regression must be 1-D [C]"
        assert reg_np.shape[0] == self.C, f"regression length {reg_np.shape[0]} != C={self.C}"

        # --- EMA smoothing ---
        if self.smoothed_reg is None:
            self.smoothed_reg = reg_np.astype(np.float32).copy()
            alpha = 1.0
        else:
            dt = (now - self.last_time) if self.last_time is not None else None
            if dt is None or dt <= 0.0:
                alpha = 1.0
            else:
                alpha = 1.0 - np.exp(-float(dt) / max(self.tau, 1e-6))
                alpha = float(np.clip(alpha, 0.0, 1.0))
            self.smoothed_reg = (1.0 - alpha) * self.smoothed_reg + alpha * reg_np

        # --- Candidate phase from smoothed reg ---
        candidate_phase = int(np.argmin(self.smoothed_reg))

        # --- Constraints ---
        if not self.seen_phase0:
            # Clamp to 0 until phase 0 is observed once
            if candidate_phase == 0:
                self.seen_phase0 = True
                self.current_phase = 0
            else:
                self.current_phase = 0
        else:
            # Monotonic non-decreasing
            if candidate_phase >= self.current_phase:
                self.current_phase = candidate_phase

        # --- TTNP from next phase's smoothed reg ---
        if self.current_phase >= self.C - 1:
            ttnp = 0.0
        else:
            ttnp = float(self.smoothed_reg[self.current_phase + 1])

        self.last_time = now
        return self.current_phase, self.smoothed_reg.copy(), ttnp


# -------------------------
# Completion EMA (optional)
# -------------------------
class EMA1D:
    """Simple time-aware EMA for a scalar stream: alpha = 1 - exp(-dt/tau)."""

    def __init__(self, tau_seconds: float):
        self.tau = float(tau_seconds)
        self.value: Optional[float] = None
        self.last_time: Optional[float] = None

    def update(self, x: float, now: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            dt = (now - self.last_time) if self.last_time is not None else None
            if dt is None or dt <= 0.0:
                alpha = 1.0
            else:
                alpha = 1.0 - np.exp(-float(dt) / max(self.tau, 1e-6))
                alpha = float(np.clip(alpha, 0.0, 1.0))
            self.value = (1.0 - alpha) * self.value + alpha * float(x)
        self.last_time = now
        return float(self.value)


# -------------------------
# ROS Node
# -------------------------
class HighLevelPolicyNode:
    def __init__(self):
        # Allow overriding ROS_MASTER_URI default used in your environment
        if "ROS_MASTER_URI" not in os.environ or "localhost" in os.environ.get("ROS_MASTER_URI", ""):
            rospy.loginfo("Setting default ROS MASTER to James")
            os.environ["ROS_MASTER_URI"] = "http://James:11311"

        rospy.init_node("anticipation_policy_node", anonymous=False)

        # --- Params ---
        self.tau_seconds = rospy.get_param("~tau_seconds", 0.6)
        self.start_phase = int(rospy.get_param("~start_phase", 0))
        self.horizon_H = float(rospy.get_param("~horizon_H", 2.0))

        self.smooth_completion = bool(rospy.get_param("~smooth_completion", False))
        self.compl_tau_seconds = float(rospy.get_param("~completion_tau_seconds", self.tau_seconds))

        topics = rospy.get_param("~topics", {})
        self.topic_reg_in = topics.get("regression", "/anticipation/regression")
        self.topic_phase_in = topics.get("phase", "/anticipation/phase")
        self.topic_completion_in = topics.get("completion", "/anticipation/completion")

        self.topic_anticip_out = topics.get("anticip_smooth", "/anticipation/anticip_smooth")
        self.topic_phase_out = topics.get("phase_smooth", "/anticipation/phase_smooth")
        self.topic_ttnp_out = topics.get("ttnp", "/anticipation/ttnp")
        self.topic_completion_out = topics.get("completion_smooth", "/anticipation/completion_smooth")

        # --- State ---
        self.C: Optional[int] = None
        self.policy: Optional[HighLevelPolicy] = None
        self.compl_ema: Optional[EMA1D] = EMA1D(self.compl_tau_seconds) if self.smooth_completion else None

        # --- Publishers ---
        self.pub_anticip = rospy.Publisher(self.topic_anticip_out, Float32MultiArray, queue_size=10)
        self.pub_phase = rospy.Publisher(self.topic_phase_out, Int32, queue_size=10)
        self.pub_ttnp = rospy.Publisher(self.topic_ttnp_out, Float32, queue_size=10)
        self.pub_completion = None
        if self.smooth_completion:
            self.pub_completion = rospy.Publisher(self.topic_completion_out, Float32, queue_size=10)

        # --- Subscribers ---
        # NOTE: regression drives the policy; phase/completion are optional inputs.
        self.sub_reg = rospy.Subscriber(self.topic_reg_in, Float32MultiArray, self.cb_regression, queue_size=10)
        self.sub_phase = rospy.Subscriber(self.topic_phase_in, Int32, self.cb_phase, queue_size=10)
        self.sub_completion = rospy.Subscriber(self.topic_completion_in, Float32, self.cb_completion, queue_size=10)

        self.last_raw_phase: Optional[int] = None  # informative only (from /anticipation/phase)

        rospy.loginfo("[anticipation_policy_node] Ready. Subscribed to RAW topics.")

    # -------- Callbacks --------
    def cb_phase(self, msg: Int32):
        # Not used for decision (policy is reg-driven), but handy for debugging/telemetry.
        self.last_raw_phase = int(msg.data)

    def cb_completion(self, msg: Float32):
        # If completion smoothing enabled, buffer latest value & publish EMA on next regression tick
        if not self.smooth_completion or self.compl_ema is None:
            return
        # Defer emission to the regression callback to keep times aligned? We can smooth immediately.
        now = rospy.get_time()
        value = float(msg.data)
        smoothed = self.compl_ema.update(value, now)
        # Publish immediately — or comment this out if you prefer only on reg updates.
        self.pub_completion.publish(Float32(data=smoothed))

    def cb_regression(self, msg: Float32MultiArray):
        now = rospy.get_time()

        reg_list = list(msg.data)
        if self.C is None:
            self.C = len(reg_list)
            self.policy = HighLevelPolicy(
                num_classes=self.C,
                horizon_H=self.horizon_H,
                tau_seconds=self.tau_seconds,
                start_phase=self.start_phase,
            )
            rospy.loginfo(f"[anticipation_policy_node] Initialized policy with C={self.C}, tau={self.tau_seconds}, H={self.horizon_H}")

        if len(reg_list) != self.C:
            rospy.logwarn_throttle(2.0, f"[anticipation_policy_node] Received regression len {len(reg_list)} != C={self.C}; ignoring.")
            return

        reg_np = np.asarray(reg_list, dtype=np.float32)

        # --- Run policy ---
        phase_proc, reg_smooth, ttnp = self.policy.update(reg_np, now)

        # --- Publish processed outputs ---
        out_reg = Float32MultiArray(data=reg_smooth.astype(np.float32).tolist())
        self.pub_anticip.publish(out_reg)
        self.pub_phase.publish(Int32(data=int(phase_proc)))
        self.pub_ttnp.publish(Float32(data=float(ttnp)))

        # (Optional) also emit completion EMA aligned with regression ticks
        if self.smooth_completion and self.compl_ema is not None and self.pub_completion is not None and self.compl_ema.value is not None:
            self.pub_completion.publish(Float32(data=float(self.compl_ema.value)))

    def spin(self):
        rospy.spin()


def main():
    node = HighLevelPolicyNode()
    node.spin()


if __name__ == "__main__":
    main()