#!/usr/bin/env python3
# train_future_anticipation_feedback.py
from math import tau
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import random
from pathlib import Path
from typing import Dict, Any, DefaultDict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets.peg_and_ring_workflow import PegAndRing, VideoBatchSampler

# ---- backbone (torchvision) ----
try:
    import torchvision
    from torchvision.models import (
        resnet18,
        resnet50,
        ResNet18_Weights,
        ResNet50_Weights,
    )

    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore

    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# ---------------------------
# Fixed hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
TIME_UNIT = "minutes"
NUM_CLASSES = 6

FPS = 1  # 1
SEQ_LEN = 30 * FPS  # T seconds
STRIDE = 1


BATCH_SIZE = (
    16 if FPS > 1 else 32
)  # MAX = 1000 - 1200 frames per batch (BATCH_SIZE * SEQ_LEN)
NUM_WORKERS = 30
EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 2e-4

TIME_HORIZON = 2.0  # clamp/regression horizon (minutes)
MIN_CLAMP_VAL = 0.0
PRINT_EVERY = 20
CKPT_PATH = Path("peg_and_ring_cnn_feedback.pth")  # required name

EVAL_ROOT = Path("results/eval_outputs_feedback")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ROOT = Path("results/train_outputs_feedback")
TRAIN_CLS_DIR = TRAIN_ROOT / "classification"
TRAIN_ANT_DIR = TRAIN_ROOT / "anticipation"
TRAIN_CLS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_ANT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def clamp_and_pad_prevA(
    prevA: torch.Tensor,  # [B, prev_T, C]
    M: int,
    C: int,
    horizon: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Ensure the previous anticipation memory has exactly M steps.
    If prev_T < M, left-pad with default prior [0, H, H, ...];
    if prev_T > M, take the last M steps.
    Returns [B, M, C].
    """
    prevA = torch.clamp(prevA.to(device), MIN_CLAMP_VAL, horizon)
    B, prev_T, C_in = prevA.shape
    assert C_in == C, f"prevA last dim {C_in} != C {C}"
    if prev_T == M:
        return prevA
    if prev_T > M:
        return prevA[:, -M:, :]
    # left-pad
    pad_row = (
        torch.tensor([0.0] + [horizon] * (C - 1), device=device)
        .view(1, 1, C)
        .expand(B, M - prev_T, C)
    )
    return torch.cat([pad_row, prevA], dim=1)


# ---------------------------
# Model
# ---------------------------
class VisualFeatureExtractor(nn.Module):
    """
    CNN feature extractor (ResNet18/50) -> global pooled vector per frame.
    """

    def __init__(
        self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = False
    ):
        super().__init__()
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Keep everything up to the last conv block; add GAP
        self.conv = nn.Sequential(*list(bb.children())[:-2])  # [B, F, h, w]
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, F, 1, 1]
        self.feat_dim = feat_dim

        if freeze:
            for p in self.conv.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*T, 3, H, W] -> [B*T, F]
        """
        fmap = self.conv(x)
        vec = self.gap(fmap).flatten(1)
        return vec


class SoftClamp(nn.Module):
    """
    Smooth clamp to [low, high] using:
      y = low + softplus(beta*(x - low))/beta - softplus(beta*(x - high))/beta
    beta is learned (scalar or per-class). Larger beta -> closer to hard clamp.
    """

    def __init__(
        self,
        low: float,
        high: float,
        C: int,
        init_beta: float = 4.0,
        per_channel: bool = True,
    ):
        super().__init__()
        assert high > low
        self.register_buffer("low", torch.tensor(float(low)))
        self.register_buffer("high", torch.tensor(float(high)))
        self.per_channel = per_channel
        init = float(np.log(init_beta))
        if per_channel:
            self.log_beta = nn.Parameter(torch.full((C,), init))
        else:
            self.log_beta = nn.Parameter(torch.tensor(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.low, self.high
        beta = self.log_beta.exp()
        if self.per_channel:
            while beta.dim() < x.dim():
                beta = beta.unsqueeze(0)
        y = (
            low
            + F.softplus(beta * (x - low)) / beta
            - F.softplus(beta * (x - high)) / beta
        )
        # safety clamp for numeric noise (doesn't kill grads in practice near interior)
        return y.clamp(min=low.item(), max=high.item())


class AnticipationTemporalModel(nn.Module):
    """
    Simple yet expressive architecture:

    1) Visual encoder (ResNet) -> per-frame features
    2) Temporal reasoner over T frames (GRU)
    3) Previous anticipation memory of length M=SEQ_LEN (from dataloader) processed by an MLP
    4) Merge temporal summary (last hidden state) with feedback embedding
    5) Two heads:
        - Phase logits (classification)
        - Time-to-next-phase per class (regression; clamped and row-min normalized)
    """

    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        future_F: int = 5,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        d_model: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        feedback_dropout: float = 0.1,
    ):
        super().__init__()
        self.T = int(sequence_length)
        self.C = int(num_classes)
        self.H = float(time_horizon)
        self.d_model = int(d_model)
        self.M = future_F
        if self.M < 1:
            self.use_prev_anticipation = False
        else:
            self.use_prev_anticipation = True

        # ---- Visual encoder ----
        self.encoder = VisualFeatureExtractor(
            backbone, pretrained_backbone, freeze_backbone
        )
        self.frame_proj = nn.Sequential(
            nn.Linear(self.encoder.feat_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- Temporal reasoner (causal) ----
        self.temporal_rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ---- Previous anticipation encoder (MLP) ----
        # Input is flattened [B, M*C]; normalize by H to help scale.
        self.prev_encoder = nn.Sequential(
            nn.Linear(self.M * self.C, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(feedback_dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # ---- Optional FiLM modulation of temporal inputs from feedback ----
        # self.film = nn.Linear(d_model, 2 * d_model)  # produces gamma, beta

        # ---- Fusion of temporal summary and feedback embedding ----
        # self.fusion = nn.Sequential(
        #     nn.Linear(2 * d_model, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        # )

        # ---- Heads ----
        self.bound_reg = SoftClamp(
            low=MIN_CLAMP_VAL, high=self.H, C=self.C, init_beta=4.0, per_channel=True
        )

        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, self.C),
        )
        self.anticip_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, self.C),
            # nn.Softplus(beta=1.0),  # positive outputs
        )

        # scale to control feedback strength if needed
        # self.fb_scale = 1.0

        self._init_weights()

    def _init_weights(self):
        for module in [
            self.frame_proj,
            self.prev_encoder,
            # self.film,
            # self.fusion,
            self.phase_head,
            self.anticip_head,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: torch.Tensor,  # [B, T, 3, H, W]
        prev_anticipation: torch.Tensor,  # [B, prev_T, C] (will be padded/truncated to M=T)
        return_aux: bool = False,
    ):
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"
        device = frames.device

        # ---- Prepare previous anticipation memory ----
        # prevA = clamp_and_pad_prevA(
        #     prev_anticipation, self.M, self.C, self.H, device
        # )  # [B, M, C]
        prevA_norm = prev_anticipation / max(self.H, 1e-6)  # scale to ~[0,1]
        prev_flat = prevA_norm.reshape(B, self.M * self.C)  # [B, M*C]
        fb_emb = self.prev_encoder(prev_flat)  # [B, d]

        # ---- Visual encoding (per-frame) ----
        x = frames.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.encoder.training):
            feats = self.encoder(x)  # [B*T, F]
        feats = self.frame_proj(feats)  # [B*T, d]
        feats = feats.view(B, T, self.d_model)  # [B, T, d]

        # ---- Add feedback embedding ----
        feats = feats + fb_emb.unsqueeze(1)

        # ---- Temporal reasoning (causal GRU) ----
        rnn_out, hN = self.temporal_rnn(feats)  # rnn_out: [B, T, d], hN: [L, B, d]
        h_last = rnn_out[:, -1, :]  # [B, d]

        # ---- Heads ----
        logits = self.phase_head(h_last)  # [B, C]
        reg_raw = self.anticip_head(h_last)  # [B, C]

        # DIRECTLY SMOOTH
        # reg = self.bound_reg(reg_raw)         # smoothly in [MIN_VAL, H]

        # V2 SMOOTH with SOFT REBOUND
        def soft_row_min(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
            # shape [B, 1]
            return -tau * torch.logsumexp(-x / tau, dim=1, keepdim=True)

        # after soft clamp
        row_min = soft_row_min(reg_raw, tau=0.05)
        reg = reg_raw - row_min  # encourage at least one class close to 0
        reg = self.bound_reg(reg)  # keep within [-1, H]

        if return_aux:
            aux = {
                "fb_emb": fb_emb.detach(),
                "h_last": h_last.detach(),
            }
            return reg, logits, aux
        return reg, logits


def dataloader_to_inference(
    model, frames, meta, device, starts, prev_anticip_pred=None
):
    frames = frames.to(device)
    video_names = meta["video_name"]
    curr_video = video_names[0]
    if curr_video not in starts:
        new_video = True
        starts[curr_video] = True
    else:
        new_video = False

    prev_anticip_gt = meta["last_anticipation"].to(device)  # [B,C]
    if new_video or prev_anticip_pred is None:
        prev_anticip_pred = torch.clamp(
            prev_anticip_gt.float(), MIN_CLAMP_VAL, TIME_HORIZON
        )
    reg, logits, _ = model(frames, prev_anticip_pred, return_aux=True)
    return reg, logits


# ---------------------------
# Eval
# ---------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, time_horizon: float
) -> Dict[str, Any]:
    model.eval()

    total_mae = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()
    starts = {}
    prev_anticip_pred = None
    for _, (frames, meta) in enumerate(loader):

        labels = meta["phase_label"].to(device).long()
        ttnp = torch.clamp(
            meta["time_to_next_phase"].to(device).float(), MIN_CLAMP_VAL, TIME_HORIZON
        )
        prev_anticip_gt = meta["last_anticipation"].to(device)  # [B,C]
        prev_anticip_gt = torch.clamp(
            prev_anticip_gt.float(), MIN_CLAMP_VAL, TIME_HORIZON
        )
        reg, logits = dataloader_to_inference(
            model, frames, meta, device, starts, prev_anticip_gt
        )

        preds_cls = logits.argmax(dim=1)
        acc = (preds_cls == labels).float().mean()
        mae = torch.mean(torch.abs(reg - ttnp))
        ce = ce_loss(logits, labels)

        bs = frames.size(0)
        total_mae += float(mae.item()) * bs
        total_ce += float(ce.item()) * bs
        total_acc += float(acc.item()) * bs
        total_samples += bs

    return {
        "mae": total_mae / max(1, total_samples),
        "ce": total_ce / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
        "samples": total_samples,
    }


# ---------------------------
# Visualizations
# ---------------------------
@torch.no_grad()
def visualize_phase_timelines_classification(
    model: nn.Module,
    root_dir: str,
    split: str = "test",
    time_unit: str = "minutes",
    batch_size: int = 64,
    num_workers: int = 6,
    out_dir: Path = EVAL_CLS_DIR,
):
    device = next(model.parameters()).device
    model.eval()

    ds = PegAndRing(
        root_dir=root_dir,
        mode=split,
        seq_len=SEQ_LEN,
        stride=1,
        time_unit=time_unit,
        fps=FPS,
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen,
        drop_last=True,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )

    lengths: Dict[str, int] = {
        vn: ds.ant_cache[vn].shape[0] for vn in ds.ant_cache.keys()
    }
    preds: DefaultDict[str, np.ndarray] = DefaultDict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = DefaultDict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = -np.ones(N, dtype=np.int32)
        gts[vn] = -np.ones(N, dtype=np.int32)

    starts = {}
    for frames, meta in loader:
        prev_anticip_gt = meta["last_anticipation"].to(device)  # [B,C]
        prev_anticip_gt = torch.clamp(
            prev_anticip_gt.float(), MIN_CLAMP_VAL, TIME_HORIZON
        )
        reg, logits = dataloader_to_inference(
            model, frames, meta, device, starts, prev_anticip_gt
        )

        pred_phase = logits.argmax(dim=1).detach().cpu().numpy()
        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()
        gt_phase = meta["phase_label"].cpu().numpy()

        for vn, idx, p, g in zip(video_names, idx_last, pred_phase, gt_phase):
            idx = int(idx)
            if 0 <= idx < preds[vn].shape[0]:
                preds[vn][idx] = int(p)
            if 0 <= idx < gts[vn].shape[0]:
                gts[vn][idx] = int(g)

    for vn, N in lengths.items():
        pred_arr = preds[vn]
        gt_arr = gts[vn]
        x = np.arange(N)
        valid = (pred_arr >= 0) & (gt_arr >= 0)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.step(
            x[valid],
            gt_arr[valid],
            where="post",
            linewidth=2,
            label="GT Phase",
            alpha=0.9,
        )
        ax.step(
            x[valid],
            pred_arr[valid],
            where="post",
            linewidth=2,
            label="Pred Phase",
            alpha=0.9,
        )

        ax.set_title(f"[{split}] Phase timeline — {vn}")
        ax.set_xlabel("Frame index (1 FPS)")
        ax.set_ylabel("Phase (0..5)")
        ax.set_yticks(list(range(NUM_CLASSES)))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        out_path = out_dir / f"{vn}_phase_timeline_{split}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved classification timeline -> {out_path}")


@torch.no_grad()
def visualize_anticipation_curves(
    model: nn.Module,
    root_dir: str,
    split: str = "test",
    time_unit: str = "minutes",
    time_horizon: float = TIME_HORIZON,
    batch_size: int = 64,
    num_workers: int = 6,
    out_dir: Path = EVAL_ANT_DIR,
):
    device = next(model.parameters()).device
    model.eval()

    ds = PegAndRing(
        root_dir=root_dir,
        mode=split,
        seq_len=SEQ_LEN,
        stride=1,
        time_unit=time_unit,
        fps=FPS,
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen,
        drop_last=True,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )

    lengths: Dict[str, int] = {
        vn: ds.ant_cache[vn].shape[0] for vn in ds.ant_cache.keys()
    }
    preds: DefaultDict[str, np.ndarray] = DefaultDict(lambda: None)
    gts: DefaultDict[str, np.ndarray] = DefaultDict(lambda: None)
    for vn, N in lengths.items():
        preds[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)
        gts[vn] = np.full((N, NUM_CLASSES), np.nan, dtype=np.float32)

    starts = {}
    for frames, meta in loader:
        prev_anticip_gt = meta["last_anticipation"].to(device)  # [B,C]
        prev_anticip_gt = torch.clamp(
            prev_anticip_gt.float(), MIN_CLAMP_VAL, TIME_HORIZON
        )
        outputs, logits = dataloader_to_inference(
            model, frames, meta, device, starts, prev_anticip_gt
        )

        pred = (
            torch.clamp(outputs, min=MIN_CLAMP_VAL, max=time_horizon)
            .detach()
            .cpu()
            .numpy()
        )
        gt = (
            torch.clamp(meta["time_to_next_phase"], min=MIN_CLAMP_VAL, max=time_horizon)
            .cpu()
            .numpy()
        )

        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].cpu().numpy()

        for vn, idx, p_row, g_row in zip(video_names, idx_last, pred, gt):
            idx = int(idx)
            if 0 <= idx < preds[vn].shape[0]:
                preds[vn][idx, :] = p_row
            if 0 <= idx < gts[vn].shape[0]:
                gts[vn][idx, :] = g_row

    unit_tag = time_unit.lower()
    for vn, N in lengths.items():
        arr = preds[vn]
        gt = gts[vn]
        valid = np.isfinite(arr).all(axis=1) & np.isfinite(gt).all(axis=1)
        x = np.arange(N)[valid]
        arr = arr[valid]
        gt = gt[valid]

        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
        y_upper = time_horizon * 1.05
        y_lower = MIN_CLAMP_VAL * 1.05

        for i in range(NUM_CLASSES):
            axs[i].plot(x, gt[:, i], linestyle="--", label="Ground Truth", linewidth=2)
            axs[i].plot(x, arr[:, i], linestyle="-", label="Predicted", linewidth=2)
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True, linestyle="--", alpha=0.4)
            axs[i].set_ylim(y_lower, y_upper)

        axs[-1].set_xlabel("Frame index (1 FPS)")
        fig.suptitle(
            f"[{split}] Time to Next Phase — {vn} ({unit_tag}, horizon={time_horizon})"
        )
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = (
            out_dir / f"{vn}_anticipation_{unit_tag}_h{time_horizon:g}_{split}.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved anticipation curves -> {out_path}")


# ---------------------------
# Train
# ---------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Datasets
    train_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="train",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        fps=FPS,
    )
    val_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        fps=FPS,
    )
    test_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="test",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
        fps=FPS,
    )

    # Dataloaders (video-aware: batches do not cross videos; windows in order)
    gen_train = torch.Generator()
    gen_train.manual_seed(SEED)
    train_batch_sampler = VideoBatchSampler(
        train_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=True,
        generator=gen_train,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        # ---------------------------------
        batch_sampler=train_batch_sampler,
        # batch_size=BATCH_SIZE,
        # shuffle=True,
        # drop_last=True,
        # ---------------------------------
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    gen_eval = torch.Generator()
    gen_eval.manual_seed(SEED)
    val_batch_sampler = VideoBatchSampler(
        val_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_batch_sampler = VideoBatchSampler(
        test_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Model
    model = AnticipationTemporalModel(
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        time_horizon=TIME_HORIZON,
        backbone="resnet18",
        pretrained_backbone=True,
        freeze_backbone=False,
        # d_model=512,
        # gru_layers=6,
        d_model=256,
        gru_layers=2,
        gru_dropout=0.2,
        feedback_dropout=0.2,
        future_F=1,
    ).to(device)

    # print number of params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        print(f"Number of trainable parameters: {num_params/1e6:.2f}M")
    elif num_params >= 1e3:
        print(f"Number of trainable parameters: {num_params/1e3:.2f}k")
    else:
        print(f"Number of trainable parameters: {num_params}")

    # Losses/Optim
    smooth_l1 = nn.SmoothL1Loss(beta=0.5)
    anticipation_weight = 1.0
    ce_loss = nn.CrossEntropyLoss()
    phase_weight = 0.2
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -float("inf")
    best_val_mae = float("inf")

    # Train
    show_fit = True
    if True:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            # optional ramp if you want to anneal feedback FiLM
            # model.fb_scale = min(1.0, 0.2 + 0.1 * (epoch - 1))

            epoch_mae = 0.0
            epoch_ce = 0.0
            epoch_loss = 0.0
            epoch_acc = 0.0
            seen = 0

            t0 = time.time()
            starts = {}

            for it, (frames, meta) in enumerate(train_loader, start=1):
                frames = frames.to(device)
                video_names = meta["video_name"]
                curr_video = video_names[0]
                if curr_video not in starts:
                    new_video = True
                    starts[curr_video] = True
                else:
                    new_video = False

                labels = meta["phase_label"].to(device).long()
                ttnp = torch.clamp(
                    meta["time_to_next_phase"].to(device).float(),
                    MIN_CLAMP_VAL,
                    TIME_HORIZON,
                )
                prev_anticip_gt = meta["last_anticipation"].to(device)  # [B,C]
                prev_anticip_gt = torch.clamp(
                    prev_anticip_gt.float(), MIN_CLAMP_VAL, TIME_HORIZON
                )
                if new_video:
                    prev_anticip_pred = prev_anticip_gt
                reg, logits, _ = model(frames, prev_anticip_pred, return_aux=True)
                # prev_anticip_pred = reg.detach()
                prev_anticip_pred = prev_anticip_gt

                phase_loss = ce_loss(logits, labels)
                anticipation_loss = smooth_l1(reg, ttnp)

                # Weight losses
                anticipation_loss = anticipation_weight * anticipation_loss
                phase_loss = phase_weight * phase_loss

                # Final loss
                train_loss = anticipation_loss + phase_loss

                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_cls = logits.argmax(dim=1)
                    train_mae = torch.mean(torch.abs(reg - ttnp))
                    train_acc = (pred_cls == labels).float().mean()

                bs = frames.size(0)
                epoch_mae += float(train_mae.item()) * bs
                epoch_ce += float(phase_loss.item()) * bs
                epoch_loss += float(train_loss.item()) * bs
                epoch_acc += float(train_acc.item()) * bs
                seen += bs

                if it % PRINT_EVERY == 0:
                    print(
                        f"[Epoch {epoch:02d} | it {it:04d}] "
                        f"acc={train_acc.item():.4f} | cls={phase_loss.item():.4f} | "
                        f"mae={train_mae.item():.4f} | reg={anticipation_loss.item():.4f} | "
                        f"loss={train_loss.item():.4f}"
                    )

            train_loss_avg = epoch_loss / max(1, seen)
            train_mae_avg = epoch_mae / max(1, seen)
            train_ce_avg = epoch_ce / max(1, seen)
            train_acc_avg = epoch_acc / max(1, seen)
            print(
                f"\nEpoch {epoch:02d} [{time.time()-t0:.1f}s] "
                f"| train_acc={train_acc_avg:.4f} train_ce={train_ce_avg:.4f} "
                f"train_mae={train_mae_avg:.4f} train_loss={train_loss_avg:.4f}"
            )

            # Validation — prioritize phase accuracy
            val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
            print(
                f"           VAL — acc={val_stats['acc']:.4f} ce={val_stats['ce']:.4f} "
                f"mae={val_stats['mae']:.4f} | samples={val_stats['samples']}"
            )

            val_mae = val_stats["mae"]
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), CKPT_PATH)
                print(
                    f"✅  New best val_mae={best_val_mae:.4f} val_acc={val_stats['acc']:.4f} — saved to: {CKPT_PATH}"
                )

            if show_fit and (epoch == EPOCHS - 1 or train_acc_avg > 0.95):
                show_fit = False  #  show just once
                print("Visualizing training results...")
                start = time.time()
                visualize_phase_timelines_classification(
                    model=model,
                    root_dir=ROOT_DIR,
                    split="train",
                    time_unit=TIME_UNIT,
                    batch_size=BATCH_SIZE,
                    num_workers=min(6, NUM_WORKERS),
                    out_dir=TRAIN_CLS_DIR,
                )
                visualize_anticipation_curves(
                    model=model,
                    root_dir=ROOT_DIR,
                    split="train",
                    time_unit=TIME_UNIT,
                    time_horizon=TIME_HORIZON,
                    batch_size=BATCH_SIZE,
                    num_workers=min(6, NUM_WORKERS),
                    out_dir=TRAIN_ANT_DIR,
                )
                print(f"Done in {time.time() - start:.1f}s")

    # Test + Visualizations
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"\nLoaded best model from: {CKPT_PATH}")

    model.eval()
    test_stats = evaluate(model, test_loader, device, TIME_HORIZON)
    print(
        f"\nTEST — acc={test_stats['acc']:.4f} ce={test_stats['ce']:.4f} "
        f"mae={test_stats['mae']:.4f} samples={test_stats['samples']}"
    )

    visualize_phase_timelines_classification(
        model=model,
        root_dir=ROOT_DIR,
        split="test",
        time_unit=TIME_UNIT,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_CLS_DIR,
    )
    visualize_anticipation_curves(
        model=model,
        root_dir=ROOT_DIR,
        split="test",
        time_unit=TIME_UNIT,
        time_horizon=TIME_HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=EVAL_ANT_DIR,
    )

    # TRAIN RES
    visualize_phase_timelines_classification(
        model=model,
        root_dir=ROOT_DIR,
        split="train",
        time_unit=TIME_UNIT,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=TRAIN_CLS_DIR,
    )
    visualize_anticipation_curves(
        model=model,
        root_dir=ROOT_DIR,
        split="train",
        time_unit=TIME_UNIT,
        time_horizon=TIME_HORIZON,
        batch_size=BATCH_SIZE,
        num_workers=min(6, NUM_WORKERS),
        out_dir=TRAIN_ANT_DIR,
    )


if __name__ == "__main__":
    main()
