import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.membank_model import MemBankResNetLSTM


def create_memorybank(
    model: MemBankResNetLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mb_path: str = "./cholec80_memory_bank.pkl",
    device: torch.device = torch.device("cuda:0"),
):

    # create memory bank
    if os.path.exists(mb_path):
        print("Loading memory bank...")
        with open(mb_path, "rb") as f:
            memory_bank = torch.load(f, weights_only=False)
            train_mb = memory_bank["train"]
            val_mb = memory_bank["val"]
        print("Memory bank loaded!")
    else:
        print("Creating memory bank...")

        # ensure train_loader is using the test transform (not the train transform)
        train_loader.dataset.transform = val_loader.dataset.transform

        train_mb = []
        val_mb = []

        # assuming already trained model and already in the correct device
        # freezing
        for params in model.parameters():
            params.requires_grad = False
        model.eval()

        total_steps = len(train_loader) + len(val_loader)
        bar = tqdm(total=total_steps)

        loaders = [("train", train_loader), ("val", val_loader)]

        with torch.no_grad():
            batch: tuple[torch.Tensor, dict[str, torch.Tensor]]
            for loader_name, curr_loader in loaders:
                if loader_name == "train":
                    mb = train_mb
                else:
                    mb = val_mb

                for batch in curr_loader:
                    inputs, meta = batch
                    labels = meta["phase_label"]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs_feature: torch.Tensor = model.get_features(inputs)
                    # outputs_feature has shape (batch_size*seq_len, 512)
                    outputs_feature = outputs_feature.detach().cpu().numpy()
                    for j in range(len(outputs_feature)):
                        mb.append(outputs_feature[j].reshape(1, 512))
                    bar.update(1)
                print(f"{loader_name} MB feature length:", len(mb))
        bar.close()
        print("finish!")
        train_mb = np.asarray(train_mb)
        val_mb = np.asarray(val_mb)
        memory_bank = {"train": train_mb, "val": val_mb}
        with open(mb_path, "wb") as f:
            torch.save(memory_bank, f)
        print("Memory bank created!")

    return train_mb, val_mb


# def get_long_feature(start_index_list, dict_start_idx_LFB, lfb, LFB_length: int = 30):
#     # lfb is the memorybank. Size is [num, 512], where num is the number of elements in the dataset
#     # dict_start_idx_LFB is a dictionary of [int,int] from 0 to num-1
#     # start_index_list is a list of int from 0 to num-1 subsampled every sequence_length
#     """
#     start_index_list: A list of integers representing the starting indices for the sequences to be extracted.
#     dict_start_idx_LFB: A dictionary mapping each index to a corresponding index in the memory bank (lfb).
#     lfb: The memory bank, which is a 2D tensor of size [num, 512], where num is the number of elements in the dataset, and each element has 512 features.
#     """
#     long_feature = []
#     for j in range(len(start_index_list)):
#         long_feature_each = []
#         # 上一个存在feature的index
#         last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]
#         for k in range(LFB_length):
#             LFB_index = (start_index_list[j] - k - 1)
#             if int(LFB_index) in dict_start_idx_LFB:
#                 LFB_index = dict_start_idx_LFB[int(LFB_index)]
#                 long_feature_each.append(lfb[LFB_index])
#                 last_LFB_index_no_empty = LFB_index
#             else:
#                 long_feature_each.append(lfb[last_LFB_index_no_empty])
#         long_feature.append(long_feature_each)
#     return long_feature


def get_long_range_feature_clip_online(
    model: torch.nn.Module, inputs: torch.Tensor, MB: torch.Tensor, MB_norm: torch.Tensor | None = None, L: int = 30
):
    ## MB is just the memory bank features. But in testing we are running online mode.
    # To compensate this, we want to find which is the sequence of T consecutive frames
    # in MB that is the most similar to the current sequence of T frames in inputs.
    # This is done by computing the cosine similarity between the current sequence of T frames
    # in inputs and all the sequences of T frames in MB.
    # Output shape # Shape: [B, L, 512]

    if not hasattr(model, "get_features"):
        raise ValueError("Model does not have get_features method.")

    # Extract features from the current sequence of inputs
    with torch.no_grad():
        current_feature: torch.Tensor = model.get_features(inputs)
    current_feature = current_feature  # .detach().cpu()  # Shape: [B, nfeats]

    if MB_norm is None:
        MB = MB.squeeze()  # Shape: [N, nfeats]
        # Generate all possible T-length sequences in MB
        B, T, C, H, W = inputs.size()
        MB_windows = MB.unfold(0, T, 1).permute(0, 2, 1)  # Shape: [N-T+1, T, F]
        # Normalize MB_windows and current_feature for cosine similarity
        MB_windows = F.normalize(MB_windows, dim=-1)  # Normalize along feature dimension
    else:
        MB_windows = MB_norm
    current_feature = F.normalize(current_feature, dim=-1)  # Shape: [B, T, F]

    # Compute cosine similarity
    # Reshape for broadcasting: [B, F] x [N-T+1, T, F] -> [B, N-T+1]
    similarities = torch.einsum("bf,ntf->bn", current_feature, MB_windows)  # Dot product along T and F

    # Find the most similar sequence in MB for each batch
    best_start_indices = similarities.argmax(dim=1)  # Shape: [B]

    # Extract L consecutive frames starting from the best start indices
    if start_idx + L > MB.size(0):
        start_idx = MB.size(0) - L
    long_range_features = torch.stack(
        [MB[start_idx : start_idx + L] for start_idx in best_start_indices]
    )  # Shape: [B, L, F]

    if inputs.device != long_range_features.device:
        long_range_features = long_range_features.to(inputs.device)
    return long_range_features


def get_long_range_feature_clip(inputs: torch.Tensor, metadata: dict[str, torch.Tensor], MB: torch.Tensor, L: int = 30):
    """
    Generate the long-range feature clip using precomputed features in MB.

    Args:
        inputs (torch.Tensor): Input tensor of shape [B, T, C, H, W].
        metadata (list of dict): Metadata for each sequence in the batch.
        MB (torch.Tensor): Memory bank of pre-extracted features of shape [N, 512].
        L (int): Length of the temporal window in seconds.

    Returns:
        torch.Tensor: Long-range feature clip of shape [B, L, 512].
    """
    B, T, C, H, W = inputs.size()  # Get batch size from inputs
    feature_dim = MB.size(-1)  # Dimensionality of the features (512)

    # Initialize the output memory bank for long-range features
    long_range_features = torch.zeros(B, L, feature_dim, device=inputs.device)

    # Populate the long-range features using frames_indexes from metadata
    for i in range(B):
        frame_indexes = metadata["frames_indexes"][i, :]  # List of frame indexes associated with MB

        # Select the last L frame indexes for long-range features
        start_idx = frame_indexes[0] - L
        if start_idx < 0:
            offset = abs(start_idx)
            start_idx = 0
        else:
            offset = 0

        idxs = list(range(start_idx, frame_indexes[0] + offset))
        assert len(idxs) == L, f"Expected {L} indexes, but got {len(idxs)}"
        mb_feats = MB[idxs, :, :].squeeze()  # Extract the features from the memory bank
        long_range_features[i, :, :] = mb_feats

    return long_range_features  # Shape: [B, L, 512]


def __test_mb_creation__():
    device = torch.device("cuda:0")
    model = MemBankResNetLSTM(10).to(device)

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return torch.randn(10, 3, 224, 224), {"phase_label": torch.tensor(1)}

    train_loader = DataLoader(FakeDataset(50), batch_size=12)
    val_loader = DataLoader(FakeDataset(10), batch_size=12)

    out = "dummy_output.pkl"
    if os.path.exists(out):
        os.remove(out)
    memory_bank = create_memorybank(model=model, train_loader=train_loader, val_loader=val_loader, mb_path=out)
    os.remove(out)


def __test_long_features__():
    # Example input dimensions
    B, T, C, H, W = 40, 10, 3, 224, 224
    N = 1000  # Total number of features in memory bank
    L = 30  # Temporal length of the long-range feature clip

    inputs = torch.randn(B, T, C, H, W)  # Random input for testing
    MB = torch.randn(N, 512)  # Precomputed memory bank features

    metadata = [
        {
            "video_name": f"video_{i}",
            "frames_filepath": [f"frame_{j}.jpg" for j in range(T)],
            "frames_indexes": list(range(i * T, i * T + T)),  # Map to MB indexes
            "phase_label": "some_label",
            "phase_label_dense": ["phase_label"] * T,
        }
        for i in range(B)
    ]

    from models.temporal_model import TemporalResNetLSTM

    model = MemBankResNetLSTM(sequence_length=10)
    model = TemporalResNetLSTM(backbone=model, sequence_length=10)
    tt = get_long_range_feature_clip_online(model, inputs, MB)

    long_range_features = get_long_range_feature_clip(inputs, metadata, MB, L=L)
    print(long_range_features.shape)  # Expected: [40, 30, 512]


if __name__ == "__main__":
    # __test_mb_creation__()
    __test_long_features__()
