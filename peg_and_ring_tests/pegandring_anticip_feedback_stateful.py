#!/usr/bin/env python3
# train_future_anticipation_feedback.py
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

SEQ_LEN = 16  # T (visual window size)
STRIDE = 1

BATCH_SIZE = 64
NUM_WORKERS = 30
EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 2e-4

TIME_HORIZON = 2.0  # clamp/regression horizon (minutes)
PRINT_EVERY = 20
CKPT_PATH = Path("peg_and_ring_cnn_feedback_stateful.pth")  # required name

EVAL_ROOT = Path("eval_outputs_stateful")
EVAL_CLS_DIR = EVAL_ROOT / "classification"
EVAL_ANT_DIR = EVAL_ROOT / "anticipation"
EVAL_CLS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ANT_DIR.mkdir(parents=True, exist_ok=True)


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
    prevA = torch.clamp(prevA.to(device), 0.0, horizon)
    B, prev_T, C_in = prevA.shape
    assert C_in == C, f"prevA last dim {C_in} != C {C}"
    if prev_T == M:
        return prevA
    if prev_T > M:
        return prevA[:, -M:, :]
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

        self.conv = nn.Sequential(*list(bb.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

        if freeze:
            for p in self.conv.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.conv(x)
        vec = self.gap(fmap).flatten(1)
        return vec


class AnticipationTemporalModel(nn.Module):
    """
    Hierarchical temporal model:
      - Frame GRU over T frames → h_last
      - Feedback MLP over previous anticipations → fb_emb
      - Context GRU runs one step per window with input Proj([h_last; fb_emb]),
        hidden is kept across windows per video by the training/eval loops.
      - Heads read the context hidden.
    """

    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 2.0,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        d_model: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        feedback_dropout: float = 0.1,
        context_layers: int = 1,
        context_dropout: float = 0.0,
    ):
        super().__init__()
        self.T = int(sequence_length)
        self.C = int(num_classes)
        self.H = float(time_horizon)
        self.d_model = int(d_model)
        self.M = self.T

        # Visual encoder
        self.encoder = VisualFeatureExtractor(
            backbone, pretrained_backbone, freeze_backbone
        )
        self.frame_proj = nn.Sequential(
            nn.Linear(self.encoder.feat_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Frame-level temporal GRU
        self.temporal_rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Previous anticipation encoder (MLP)
        self.prev_encoder = nn.Sequential(
            nn.Linear(self.M * self.C, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(feedback_dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Project window summary + feedback to context input size
        self.win_to_ctx = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Context GRU (one step per window)
        self.context_rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=context_layers,
            batch_first=True,
            dropout=context_dropout if context_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Heads operate on context hidden
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.C),
        )
        self.anticip_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.C),
            nn.Softplus(beta=1.0),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [
            self.frame_proj,
            self.prev_encoder,
            self.win_to_ctx,
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
        frames: torch.Tensor,              # [B, T, 3, H, W]
        prev_anticipation: torch.Tensor,   # [B, prev_T, C]
        *,
        context_h0: Optional[torch.Tensor] = None,  # [L, B, d] or None
        return_aux: bool = False,
    ):
        B, T, C_in, H, W = frames.shape
        assert T == self.T, f"Expected sequence length {self.T}, got {T}"
        device = frames.device

        # Encode prev anticipations
        prevA = clamp_and_pad_prevA(prev_anticipation, self.M, self.C, self.H, device)
        prevA_norm = prevA / max(self.H, 1e-6)
        fb_emb = self.prev_encoder(prevA_norm.reshape(B, self.M * self.C))

        # Frame-wise visual encoding
        x = frames.view(B * T, C_in, H, W)
        with torch.set_grad_enabled(self.encoder.training):
            feats = self.encoder(x)
        feats = self.frame_proj(feats).view(B, T, self.d_model)

        # Frame GRU over the window
        rnn_out, _ = self.temporal_rnn(feats)
        h_last = rnn_out[:, -1, :]  # [B, d]

        # One-step context update
        ctx_in = self.win_to_ctx(torch.cat([h_last, fb_emb], dim=1)).unsqueeze(1)  # [B,1,d]
        if context_h0 is None:
            context_h0 = torch.zeros(
                self.context_rnn.num_layers, B, self.d_model, device=device
            )
        ctx_out, ctx_hN = self.context_rnn(ctx_in, context_h0)  # ctx_hN: [L,B,d]
        ctx_feat = ctx_out[:, -1, :]  # [B, d]

        logits = self.phase_head(ctx_feat)
        reg_raw = self.anticip_head(ctx_feat)

        reg = torch.clamp(reg_raw, 0.0, self.H)
        reg = reg - reg.min(dim=1, keepdim=True)[0]
        reg = torch.clamp(reg, 0.0, self.H)

        if return_aux:
            aux = {
                "fb_emb": fb_emb.detach(),
                "h_last": h_last.detach(),
                "ctx_hN": ctx_hN.detach(),
            }
            return reg, logits, aux
        return reg, logits, ctx_hN  # keep ctx_hN available if caller wants it


# ---------------------------
# Per-video context helpers
# ---------------------------
class ContextBank:
    """
    Keeps GRU hidden states per video across windows.
    Resets when a new video starts or when an index goes backward.
    """

    def __init__(self, num_layers: int, d_model: int, device: torch.device):
        self.num_layers = num_layers
        self.d_model = d_model
        self.device = device
        self.h: Dict[str, torch.Tensor] = {}
        self.last_idx: Dict[str, int] = {}

    def prepare_h0(
        self, video_names: List[str], last_indexes: torch.Tensor
    ) -> torch.Tensor:
        B = len(video_names)
        h0 = torch.zeros(self.num_layers, B, self.d_model, device=self.device)
        for b, vn in enumerate(video_names):
            idx = int(last_indexes[b])
            if (vn not in self.h) or (vn in self.last_idx and idx < self.last_idx[vn]):
                self.h[vn] = torch.zeros(self.num_layers, 1, self.d_model, device=self.device)
            h0[:, b : b + 1, :] = self.h[vn]
            self.last_idx[vn] = idx
        return h0

    def update(
        self, video_names: List[str], ctx_hN: torch.Tensor
    ):
        for b, vn in enumerate(video_names):
            self.h[vn] = ctx_hN[:, b : b + 1, :].detach()


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

    # context across windows during evaluation
    ctx_bank = ContextBank(
        num_layers=model.context_rnn.num_layers, d_model=model.d_model, device=device
    )

    for _, (frames, meta) in enumerate(loader):
        frames = frames.to(device)
        ttnp = torch.clamp(
            meta["time_to_next_phase"].to(device).float(), 0.0, time_horizon
        )
        labels = meta["phase_label"].to(device).long()
        prevA = meta["prev_anticipation"].to(device)

        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].to(device)

        context_h0 = ctx_bank.prepare_h0(video_names, idx_last)
        reg, logits, ctx_hN = model(
            frames, prevA, context_h0=context_h0, return_aux=False
        )
        ctx_bank.update(video_names, ctx_hN)

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
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen,
        drop_last=True
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

    ctx_bank = ContextBank(
        num_layers=model.context_rnn.num_layers, d_model=model.d_model, device=device
    )

    for frames, meta in loader:
        frames = frames.to(device)
        prevA = meta["prev_anticipation"].to(device)

        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].to(device)
        context_h0 = ctx_bank.prepare_h0(video_names, idx_last)

        _, logits, ctx_hN = model(frames, prevA, context_h0=context_h0, return_aux=False)
        ctx_bank.update(video_names, ctx_hN)

        pred_phase = logits.argmax(dim=1).detach().cpu().numpy()
        idx_last_np = meta["frames_indexes"][:, -1].cpu().numpy()
        gt_phase = meta["phase_label"].cpu().numpy()

        for vn, idx, p, g in zip(video_names, idx_last_np, pred_phase, gt_phase):
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
        root_dir=root_dir, mode=split, seq_len=SEQ_LEN, stride=1, time_unit=time_unit
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)
    batch_sampler = VideoBatchSampler(
        ds,
        batch_size=batch_size,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen,
        drop_last=True
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

    ctx_bank = ContextBank(
        num_layers=model.context_rnn.num_layers, d_model=model.d_model, device=device
    )

    for frames, meta in loader:
        frames = frames.to(device)
        prevA = meta["prev_anticipation"].to(device)

        video_names = meta["video_name"]
        idx_last = meta["frames_indexes"][:, -1].to(device)
        context_h0 = ctx_bank.prepare_h0(video_names, idx_last)

        outputs, _, ctx_hN = model(frames, prevA, context_h0=context_h0, return_aux=False)
        ctx_bank.update(video_names, ctx_hN)

        pred = torch.clamp(outputs, min=0.0, max=time_horizon).detach().cpu().numpy()
        gt = (
            torch.clamp(meta["time_to_next_phase"], min=0.0, max=time_horizon)
            .cpu()
            .numpy()
        )

        idx_last_np = meta["frames_indexes"][:, -1].cpu().numpy()

        for vn, idx, p_row, g_row in zip(video_names, idx_last_np, pred, gt):
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

        for i in range(NUM_CLASSES):
            axs[i].plot(x, arr[:, i], linestyle="-", label="Predicted", linewidth=2)
            axs[i].plot(x, gt[:, i], linestyle="--", label="Ground Truth", linewidth=2)
            axs[i].set_ylabel(f"Phase {i}")
            axs[i].grid(True, linestyle="--", alpha=0.4)
            axs[i].set_ylim(0, y_upper)

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
    )
    val_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="val",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
    )
    test_ds = PegAndRing(
        root_dir=ROOT_DIR,
        mode="test",
        seq_len=SEQ_LEN,
        stride=STRIDE,
        time_unit=TIME_UNIT,
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
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    gen_eval = torch.Generator()
    gen_eval.manual_seed(SEED)
    val_batch_sampler = VideoBatchSampler(
        val_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_batch_sampler = VideoBatchSampler(
        test_ds,
        batch_size=BATCH_SIZE,
        batch_videos=False,
        shuffle_videos=False,
        generator=gen_eval,
        drop_last=True
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
        d_model=256,
        gru_layers=2,
        gru_dropout=0.1,
        feedback_dropout=0.1,
        context_layers=1,
        context_dropout=0.0,
    ).to(device)

    # Losses/Optim
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -float("inf")

    # Train
    if True:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            # model.fb_scale = min(1.0, 0.2 + 0.1 * (epoch - 1))

            epoch_mae = 0.0
            epoch_ce = 0.0
            epoch_loss = 0.0
            epoch_acc = 0.0
            seen = 0

            t0 = time.time()

            # new context bank each epoch (no leakage)
            ctx_bank = ContextBank(
                num_layers=model.context_rnn.num_layers,
                d_model=model.d_model,
                device=device,
            )

            for it, (frames, meta) in enumerate(train_loader, start=1):
                frames = frames.to(device)
                labels = meta["phase_label"].to(device).long()
                ttnp = torch.clamp(
                    meta["time_to_next_phase"].to(device).float(), 0.0, TIME_HORIZON
                )
                prevA = meta["prev_anticipation"].to(device)
                video_names = meta["video_name"]
                idx_last = meta["frames_indexes"][:, -1].to(device)

                context_h0 = ctx_bank.prepare_h0(video_names, idx_last)

                reg, logits, aux = model(
                    frames, prevA, context_h0=context_h0, return_aux=True
                )
                ctx_bank.update(video_names, aux["ctx_hN"])

                phase_loss = ce_loss(logits, labels)
                anticipation_loss = smooth_l1(reg, ttnp)
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

            # Validation — stateful across windows per video
            val_stats = evaluate(model, val_loader, device, TIME_HORIZON)
            print(
                f"           VAL — acc={val_stats['acc']:.4f} ce={val_stats['ce']:.4f} "
                f"mae={val_stats['mae']:.4f} | samples={val_stats['samples']}"
            )

            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                torch.save(model.state_dict(), CKPT_PATH)
                print(
                    f"✅  New best val_acc={best_val_acc:.4f} — saved to: {CKPT_PATH}"
                )

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


if __name__ == "__main__":
    main()
