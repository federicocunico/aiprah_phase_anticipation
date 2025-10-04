#!/usr/bin/env python3
# train_phase_classifier_with_viz.py

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchvision.models import resnet18
from PIL import Image

from datasets.peg_and_ring_workflow import PegAndRing


# ---------------------------
# Fixed training hyperparams
# ---------------------------
SEED = 42
ROOT_DIR = "data/peg_and_ring_workflow"
SEQ_LEN = 16
STRIDE = 1
TIME_UNIT = "minutes"        # dataset outputs in minutes (or "seconds")
NUM_CLASSES = 6

BATCH_SIZE = 32
NUM_WORKERS = 31
EPOCHS = 20

LR = 1e-3
WEIGHT_DECAY = 1e-4
STEP_SIZE = 8
GAMMA = 0.1

PRINT_EVERY = 50  # iterations
OUT_DIR = Path("outputs_phase_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(loader):
        frames, meta = batch
        # frames: [B, T, 3, H, W]
        x = frames[:, -1, ...].to(device)  # last frame only
        y = meta["phase_label"].to(device).long()

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_samples += x.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "acc": acc, "samples": total_samples}


@torch.no_grad()
def _list_frames_for_video(frames_dir: Path) -> List[Path]:
    # Gather 1fps frames sorted naturally: 00000.png, 00001.png, ...
    def _natkey(s: str):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    frames = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda x: _natkey(x.name)
    )
    return frames


@torch.no_grad()
def visualize_temporal_consistency(
    model: nn.Module,
    ds_test: PegAndRing,
    device: torch.device,
    out_dir: Path = OUT_DIR,
    batch_eval: int = 128,
):
    """
    For each test video:
      - Predict phase for EVERY frame (1 FPS) using ds_test.test_transform
      - Build GT phase per frame from ds_test.ant_cache (phase = first column with 0 or argmin)
      - Save a step plot: Predicted vs Ground Truth (y in {0..5}) across frame index
    """
    model.eval()

    # Build mapping: videoname -> frames_dir (root/<videoname>_1fps)
    # We can infer available test videos from ds_test.windows
    videos: Dict[str, Path] = {}
    for w in ds_test.windows:
        vn = w["video_name"]
        if vn not in videos:
            videos[vn] = Path(ds_test.root / f"{vn}_1fps")

    # Use the dataset's deterministic test transform for preprocessing
    test_tf = ds_test.test_transform

    overall_correct = 0
    overall_total = 0

    for videoname, frames_dir in videos.items():
        if not frames_dir.exists():
            print(f"[WARN] Missing frames dir for {videoname}: {frames_dir}")
            continue

        # Load all frame paths
        frame_paths = _list_frames_for_video(frames_dir)
        N = len(frame_paths)
        if N == 0:
            print(f"[WARN] No frames in {frames_dir}")
            continue

        # Ground-truth phase per frame from ant_cache (already in requested unit for anticipation,
        # but here we only need the current phase index)
        # ds_test.ant_cache[videoname] shape is (N,6) for 1 fps frames
        if videoname not in ds_test.ant_cache:
            print(f"[WARN] No ant_cache for {videoname}")
            continue
        gt_mat = ds_test.ant_cache[videoname]
        if gt_mat.shape[0] < N:
            N = gt_mat.shape[0]
            frame_paths = frame_paths[:N]
        gt_phase = []
        for i in range(N):
            row = gt_mat[i]  # [6]
            zeros = np.where(row == 0.0)[0]
            gt_phase.append(int(zeros[0]) if len(zeros) else int(np.argmin(row)))
        gt_phase = np.asarray(gt_phase, dtype=np.int64)

        # Predict phase per frame (batch for speed)
        preds: List[int] = []
        idx = 0
        while idx < N:
            chunk = frame_paths[idx: idx + batch_eval]
            imgs = [test_tf(Image.open(p).convert("RGB")) for p in chunk]
            x = torch.stack(imgs, dim=0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            idx += len(chunk)
        preds = np.asarray(preds, dtype=np.int64)

        # Accuracy per video
        correct = int((preds == gt_phase).sum())
        acc = correct / max(1, N)
        overall_correct += correct
        overall_total += N
        print(f"[VIZ] {videoname}: N={N} acc={acc:.4f}")

        # Plot temporal consistency: step plots over frame index
        x = np.arange(N)
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.step(x, gt_phase, where="post", linewidth=2, label="GT", color="tab:red")
        ax.step(x, preds,   where="post", linewidth=2, label="Pred", color="tab:blue", alpha=0.8)
        ax.set_xlabel("Frame index (1 FPS)")
        ax.set_ylabel("Phase ID")
        ax.set_yticks(list(range(NUM_CLASSES)))
        ax.set_ylim(-0.5, NUM_CLASSES - 0.5)
        ax.grid(True, linestyle=":")
        ax.set_title(f"Temporal Phase Consistency — {videoname} (acc={acc:.3f})")
        ax.legend(loc="upper right")
        out_path = out_dir / f"{videoname}_temporal_phase.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved temporal plot -> {out_path}")

    if overall_total > 0:
        print(f"\n[TEST VIS] Overall temporal accuracy across videos: {overall_correct/overall_total:.4f} "
              f"({overall_correct}/{overall_total})")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Datasets & Dataloaders
    # ---------------------------
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

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    # ---------------------------
    # Model
    # ---------------------------
    model = resnet18(weights=None)  # Use random init; dataset already normalized to your stats
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    # ---------------------------
    # Optimizer / Scheduler / Loss
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # ---------------------------
    # Training loop
    # ---------------------------
    best_val_acc = 0.0
    save_path = OUT_DIR / "best_phase_cls_resnet18.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        start_time = time.time()

        for it, batch in enumerate(train_loader, start=1):
            frames, meta = batch
            x = frames[:, -1, ...].to(device)  # last frame
            y = meta["phase_label"].to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == y).sum().item()

            bs = x.size(0)
            epoch_loss += float(loss.item()) * bs
            epoch_correct += int(correct)
            epoch_samples += bs

            global_step += 1
            if it % PRINT_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch:02d} | it {it:04d}] "
                    f"loss={loss.item():.4f} acc={correct/bs:.4f} lr={lr:.3e}"
                )

        scheduler.step()

        # Train epoch stats
        train_loss = epoch_loss / max(1, epoch_samples)
        train_acc = epoch_correct / max(1, epoch_samples)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch:02d} done in {elapsed:.1f}s | train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        val_stats = evaluate(model, val_loader, device)
        print(
            f"           val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f} "
            f"| samples={val_stats['samples']}"
        )

        # Save best
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save(model.state_dict(), save_path)
            print(f"✅  New best val_acc={best_val_acc:.4f} — saved to: {save_path}")

    # ---------------------------
    # Test (load best)
    # ---------------------------
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"\nLoaded best model from: {save_path}")

    test_stats = evaluate(model, test_loader, device)
    print(f"\nTEST — loss={test_stats['loss']:.4f} acc={test_stats['acc']:.4f} samples={test_stats['samples']}")

    # ---------------------------
    # Final visualization on TEST set
    # ---------------------------
    visualize_temporal_consistency(model, test_ds, device, OUT_DIR)


if __name__ == "__main__":
    main()
