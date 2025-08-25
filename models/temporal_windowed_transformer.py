"""
Window-Memory Transformer — forward adapted to your VideoBatchSampler
--------------------------------------------------------------------
What changed:
- You said you'll ALWAYS use batches confined within a single video (VideoBatchSampler with batch_videos=False).
- We now accept your batch `meta` directly and *derive* `video_ids` and `is_new_video` inside the model:
    * video_ids: mapped from `meta["video_name"]` (string) to a stable integer id
    * is_new_video: True for the first window of that video in the epoch, computed by tracking
      the last seen frame index per video and checking continuity vs the current window.

- The model keeps small maps:
    * `self._vid_map: Dict[str, int]`        -> string name -> int id
    * `self._last_frame_idx: Dict[int, int]` -> last *end* index observed for each video

- Call `model.reset_all_memory()` at the end of each epoch (clears memory + continuity trackers).

Outputs:
- Regression [B, C] in [0, time_horizon] with prior ensuring ≥1 zero per row (configurable).
- Classification logits [B, C], computed on GRU over recent windows for that video.

Demo at the bottom shows how to call `forward_with_meta(frames, meta)` (your dataloader style).
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Literal, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- lightweight pretrained 2D backbone ----
try:
    import torchvision
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS_ENUM = True
except Exception:
    import torchvision  # type: ignore
    _HAS_TORCHVISION_WEIGHTS_ENUM = False


# ---------------------- Priors / projections ----------------------
def rowmin_zero_hard(y: torch.Tensor, H: float) -> torch.Tensor:
    # hard projection: subtract row-min, clamp to [0, H]
    return torch.clamp(y - y.min(dim=1, keepdim=True).values, 0.0, H)

def rowmin_zero_soft(y: torch.Tensor, H: float, tau: float = 0.2) -> torch.Tensor:
    # soft projection: subtract softmin with temperature tau
    w = F.softmax(-y / max(tau, 1e-6), dim=1)
    soft_min = (w * y).sum(dim=1, keepdim=True)
    return torch.clamp(y - soft_min, 0.0, H)

def gumbel_softmax_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    g = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + g) / tau, dim=1)
    idx = y_soft.argmax(dim=1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(1, idx, 1.0)
    return y_hard + (y_soft - y_soft.detach())

def rowmin_zero_gated(y: torch.Tensor, logits: torch.Tensor, H: float, tau: float = 0.7) -> torch.Tensor:
    s = gumbel_softmax_onehot(logits, tau=tau)   # [B,C] straight-through one-hot
    offset = (y * s).sum(dim=1, keepdim=True)    # value at selected class
    return torch.clamp(y - offset, 0.0, H)


# ---------------------- Per-video ring buffer ----------------------
class RingBufferMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffers: Dict[int, deque] = {}

    def reset(self, vid: int):
        self.buffers[vid] = deque(maxlen=self.capacity)

    def clear(self):
        self.buffers.clear()

    def push(self, vid: int, emb: torch.Tensor):
        if vid not in self.buffers:
            self.buffers[vid] = deque(maxlen=self.capacity)
        self.buffers[vid].append(emb.detach().cpu())

    def get_sequences(self, video_ids: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        B = video_ids.size(0)
        lists: List[List[torch.Tensor]] = []
        D = None
        lengths = []
        for i in range(B):
            vid = int(video_ids[i].item())
            hist = list(self.buffers.get(vid, []))
            lengths.append(len(hist))
            lists.append(hist)
            if D is None and len(hist) > 0:
                D = hist[0].numel()
        Lmax = max([len(x) for x in lists]) if lists else 0
        if Lmax == 0:
            return torch.empty(B, 0, 1, device=device), torch.zeros(B, dtype=torch.long, device=device)
        if D is None: D = 1
        out = torch.zeros(B, Lmax, D, device=device)
        for i, hist in enumerate(lists):
            for t, v in enumerate(hist):
                out[i, Lmax - len(hist) + t] = v.to(device)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        return out, lengths_t


# ---------------------- Model ----------------------
class WindowMemoryTransformer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 6,
        time_horizon: float = 1.0,
        *,
        backbone: str = "resnet18",           # "resnet18" | "resnet50"
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        reasoning_time: int = 3,              # windows to remember per video
        prior_mode: Literal["none", "hard", "soft", "gated"] = "hard",
        soft_tau: float = 0.2,
        gated_tau: float = 0.7,
        reg_activation: Literal["softplus", "linear"] = "softplus",
        stride_hint: int = 1,                 # expected window stride (for continuity check)
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.T = int(sequence_length)
        self.num_classes = int(num_classes)
        self.time_horizon = float(time_horizon)
        self.d_model = int(d_model)
        self.reasoning_time = int(reasoning_time)
        self.prior_mode = prior_mode
        self.soft_tau = float(soft_tau)
        self.gated_tau = float(gated_tau)
        self.reg_activation = reg_activation
        self.stride_hint = int(stride_hint)

        # --- backbone ---
        if backbone == "resnet18":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet18(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 512
        elif backbone == "resnet50":
            if _HAS_TORCHVISION_WEIGHTS_ENUM:
                bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
            else:
                bb = torchvision.models.resnet50(pretrained=pretrained_backbone)  # type: ignore
            feat_dim = 2048
        else:
            raise ValueError(backbone)
        bb.fc = nn.Identity()
        self.backbone = bb
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- projection + norm ---
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.feat_norm = nn.LayerNorm(d_model)

        # --- learned positional embeddings (over frames) ---
        self.pos_emb = nn.Embedding(self.T, d_model)

        # --- temporal transformer (within window) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # --- class-query decoder over current window ---
        self.class_queries = nn.Parameter(torch.randn(self.num_classes, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # --- heads ---
        if reg_activation == "softplus":
            self.reg_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus(beta=1.0))
        else:
            self.reg_head = nn.Linear(d_model, 1)

        self.cls_fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(d_model, self.num_classes)

        # --- state machine over recent windows ---
        self.window_gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.memory = RingBufferMemory(capacity=self.reasoning_time)

        # --- bookkeeping to auto-derive video_ids & is_new ---
        self._vid_map: Dict[str, int] = {}        # video_name -> int id
        self._last_frame_idx: Dict[int, int] = {} # int id -> last *end* index seen

        # init
        nn.init.xavier_uniform_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)
        if isinstance(self.reg_head, nn.Linear):
            nn.init.xavier_uniform_(self.reg_head.weight); nn.init.zeros_(self.reg_head.bias)
        nn.init.xavier_uniform_(self.cls_head.weight); nn.init.zeros_(self.cls_head.bias)

    # ---- public: reset all memory & trackers at epoch end ----
    def reset_all_memory(self):
        self.memory.clear()
        self._last_frame_idx.clear()

    # ---- helpers: encode window ----
    def _encode_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, 3, H, W]
        returns:
          winpool: [B, D]
          percls : [B, C, D]
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)                             # tokenization/patching via CNN
        with torch.set_grad_enabled(self.backbone.training):
            feats = self.backbone(x_flat)                           # [B*T, F]
        feats = feats.view(B, T, -1)
        feats = self.feat_proj(feats)                               # projection to model dim
        feats = self.feat_norm(feats)                               # normalization before attention

        pos = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(1)   # temporal positions
        src = feats.transpose(0, 1).contiguous() + pos                      # [T,B,D]

        memory = self.encoder(src)                                   # attention across time (bidirectional)

        tgt = self.class_queries.unsqueeze(1).expand(self.num_classes, B, -1).contiguous()
        percls = self.decoder(tgt, memory).transpose(0, 1)           # [B,C,D] per-class attention
        winpool = memory.mean(dim=0)                                  # [B,D] pooled window embedding
        return winpool, percls

    # ---- helpers: aggregate per-video recent windows ----
    def _aggregate_recent(
        self,
        video_ids: torch.Tensor,   # [B]
        winpool: torch.Tensor,     # [B,D]
        is_new_video: torch.Tensor # [B] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = winpool.shape
        device = winpool.device

        # per-sample reset
        for i in range(B):
            vid = int(video_ids[i].item())
            if bool(is_new_video[i].item()):
                self.memory.reset(vid)

        # history BEFORE appending current
        seqs, lengths = self.memory.get_sequences(video_ids, device=device)  # [B,Lmax,D], [B]
        if seqs.numel() == 0:
            combined = winpool.unsqueeze(1)                                  # [B,1,D]
            lengths_plus = torch.ones(B, dtype=torch.long, device=device)
        else:
            _, Lmax, _ = seqs.shape
            combined = torch.zeros(B, Lmax + 1, D, device=device)
            if Lmax > 0:
                combined[:, :Lmax] = seqs
            combined[:, -1] = winpool
            lengths_plus = torch.clamp(lengths + 1, min=1, max=Lmax + 1)

        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths_plus.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.window_gru(packed)                                      # [1,B,D]
        agg_state = h_n[-1]                                                   # [B,D]

        # push current for next time
        for i in range(B):
            vid = int(video_ids[i].item())
            self.memory.push(vid, winpool[i])

        return agg_state, lengths

    # ---- helpers: apply prior on regression ----
    def _apply_prior(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.prior_mode == "none":
            return torch.clamp(y, 0.0, self.time_horizon)
        if self.prior_mode == "hard":
            return rowmin_zero_hard(y, self.time_horizon)
        if self.prior_mode == "soft":
            return rowmin_zero_soft(y, self.time_horizon, tau=self.soft_tau)
        if self.prior_mode == "gated":
            return rowmin_zero_gated(y, logits, self.time_horizon, tau=self.gated_tau)
        raise ValueError(self.prior_mode)

    # ---- NEW: forward that accepts your dataloader's meta dict directly ----
    def forward(self, frames: torch.Tensor, meta: Dict[str, Any], return_aux: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        frames: [B, T, 3, H, W]
        meta keys (from your PegAndRing + VideoBatchSampler):
          - "video_name": list[str] or str (same name repeated within the batch)
          - "frames_indexes": LongTensor [B, T] (per-window frame indices)
          - ... (other keys are ignored here)
        We derive:
          video_ids    : LongTensor [B]
          is_new_video : BoolTensor [B]
        """
        device = frames.device
        B, T = frames.shape[0], frames.shape[1]

        # Parse video_name; collate may give list[str] (length B) or a single str
        vnames: List[str]
        if isinstance(meta["video_name"], (list, tuple)):
            # some collates return list of identical names within a batch
            vnames = [str(v) for v in meta["video_name"]]
        else:
            # single string for the whole batch → repeat B times
            vnames = [str(meta["video_name"])] * B

        # Map to stable integer ids
        vids_list: List[int] = []
        for name in vnames:
            if name not in self._vid_map:
                self._vid_map[name] = len(self._vid_map)  # assign new id
            vids_list.append(self._vid_map[name])
        video_ids = torch.tensor(vids_list, dtype=torch.long, device=device)  # [B]

        # Derive is_new_video via continuity check:
        # We consider the window's last frame index (end of window) for continuity.
        frames_idx = meta["frames_indexes"].to(device)  # [B, T]
        end_idx = frames_idx[:, -1]                     # [B]
        start_idx = frames_idx[:, 0]                    # [B]

        is_new_list: List[bool] = []
        for i in range(B):
            vid = int(video_ids[i].item())
            if vid not in self._last_frame_idx:
                # never seen this video in this epoch -> new
                is_new_list.append(True)
            else:
                # continuity: previous end should be exactly start - stride_hint
                prev_end = self._last_frame_idx[vid]
                is_new_list.append(not (start_idx[i].item() == prev_end + self.stride_hint))
            # update tracker with *current end* (we update here so the same batch order is accounted for)
            self._last_frame_idx[vid] = int(end_idx[i].item())

        is_new_video = torch.tensor(is_new_list, dtype=torch.bool, device=device)  # [B]

        # --- usual forward path using our derived ids/flags ---
        winpool, percls = self._encode_window(frames)           # [B,D], [B,C,D]
        dist_pre = (self.reg_head(percls) if isinstance(self.reg_head, nn.Sequential)
                    else self.reg_head(percls)).squeeze(-1)     # [B,C]  (>=0 if softplus; raw if linear)

        agg_state, hist_len = self._aggregate_recent(video_ids, winpool, is_new_video)
        z = self.cls_fuse(agg_state)
        logits = self.cls_head(z)                                # [B,C]

        reg = self._apply_prior(dist_pre, logits)                # [B,C] in [0,H]

        if return_aux:
            # optional: when hist_len==0, you can bias using meta["phase_label"] if desired
            aux = {"hist_len": hist_len, "video_ids": video_ids, "is_new_video": is_new_video}
            return reg, logits, aux
        return reg, logits

    # ---- legacy forward (manual ids/flags) remains available if you want to pass them yourself ----
    def _forward(
        self,
        frames: torch.Tensor,
        video_ids: Optional[torch.Tensor] = None,
        is_new_video: Optional[torch.Tensor] = None,
        initial_phase: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Keep for backward compatibility; if meta-free call is used, we expect ids/flags provided.
        winpool, percls = self._encode_window(frames)
        dist_pre = (self.reg_head(percls) if isinstance(self.reg_head, nn.Sequential)
                    else self.reg_head(percls)).squeeze(-1)
        if video_ids is None:
            video_ids = torch.zeros(frames.size(0), dtype=torch.long, device=frames.device)
        if is_new_video is None:
            is_new_video = torch.zeros(frames.size(0), dtype=torch.bool, device=frames.device)
        agg_state, hist_len = self._aggregate_recent(video_ids, winpool, is_new_video)
        z = self.cls_fuse(agg_state)
        logits = self.cls_head(z)
        reg = self._apply_prior(dist_pre, logits)
        if initial_phase is not None:
            boost = (hist_len == 0).float().unsqueeze(1)
            logits = logits + boost * F.one_hot(initial_phase.long(), num_classes=self.num_classes).float() * 2.0
        aux = {"hist_len": hist_len, "video_ids": video_ids, "is_new_video": is_new_video}
        return reg, logits, aux


# ---------------------- Minimal runnable demo (CPU) ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Config
    BATCH = 4
    T = 6
    C = 6
    H = W = 224
    TIME_H = 1.0

    model = WindowMemoryTransformer(
        sequence_length=T,
        num_classes=C,
        time_horizon=TIME_H,
        backbone="resnet18",
        pretrained_backbone=False,   # set True for training
        freeze_backbone=False,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        reasoning_time=3,
        prior_mode="hard",
        reg_activation="softplus",
        stride_hint=1,               # your dataset uses stride=1
    ).to(device)
    model.train()

    # Fake batch shaped like your DataLoader output
    def make_meta(vname: str, start: int) -> Dict[str, Any]:
        frames_indexes = torch.arange(start, start + T).unsqueeze(0).repeat(BATCH, 1)  # [B,T]
        # emulate your collate: some collates give a single string, some a list of strings
        meta = {
            "video_name": [vname] * BATCH,               # could also be just vname
            "frames_indexes": frames_indexes,
            "phase_label": torch.randint(0, C, (BATCH,)),# unused here
            "time_to_next_phase": torch.rand(BATCH, C),  # unused here
        }
        return meta

    # Two consecutive batches from the same video (should maintain continuity)
    frames1 = torch.randn(BATCH, T, 3, H, W)
    meta1 = make_meta("video08", start=0)
    reg1, logits1, aux1 = model.forward(frames1, meta1)
    print("hist_len (batch 1):", aux1["hist_len"].tolist(), "is_new:", aux1["is_new_video"].tolist())

    frames2 = torch.randn(BATCH, T, 3, H, W)
    meta2 = make_meta("video08", start=T)  # next window starts at previous end+1 (stride=1)
    reg2, logits2, aux2 = model.forward(frames2, meta2)
    print("hist_len (batch 2):", aux2["hist_len"].tolist(), "is_new:", aux2["is_new_video"].tolist())

    # New video -> reset should trigger automatically
    frames3 = torch.randn(BATCH, T, 3, H, W)
    meta3 = make_meta("video04", start=0)
    reg3, logits3, aux3 = model.forward(frames3, meta3, return_aux=True)
    print("hist_len (batch 3 new video):", aux3["hist_len"].tolist(), "is_new:", aux3["is_new_video"].tolist())

    # Check shapes & prior
    assert reg1.shape == (BATCH, C) and logits1.shape == (BATCH, C)
    assert (reg1 >= -1e-5).all() and (reg1 <= TIME_H + 1e-5).all()
    print("Row mins (hard prior):", reg1.min(dim=1).values)

    # End of epoch → clear all memory/trackers
    model.reset_all_memory()
    frames4 = torch.randn(BATCH, T, 3, H, W)
    meta4 = make_meta("video08", start=0)
    _, _, aux4 = model.forward(frames4, meta4)
    print("After reset_all_memory -> is_new:", aux4["is_new_video"].tolist())
