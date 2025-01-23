import pickle
import time
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from trainer import PhaseAnticipationTrainer
from models.temporal_model_v4 import TemporalAnticipationModel
from datasets.cholec80 import Cholec80Dataset

def train_model():
    torch.set_float32_matmul_precision("high")  # 'high' or 'medium'; for RTX GPUs
    seq_len = 30  # 10
    target_fps = 1
    num_classes = 7
    device = torch.device("cuda:0")

    demo_dataset = Cholec80Dataset(root_dir="./data/cholec80", mode="demo_train", seq_len=seq_len, fps=target_fps)
    # demo_loader = DataLoader(demo_dataset, batch_size=1, shuffle=False)

    model = TemporalAnticipationModel(time_horizon=5, sequence_length=seq_len, num_classes=num_classes)
    pl_model = PhaseAnticipationTrainer(model=model, loss_criterion=None)

    # load pretrained-checkpoint
    final_model = "./cholec80/1fw3qjzl/checkpoints/stage2_model_best.ckpt"
    pl_model.load_state_dict(torch.load(final_model, weights_only=True)["state_dict"])

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = model.to(device=device, dtype=dtype)

    # init model
    model(torch.randn(1, seq_len, 3, 224, 224).to(device=device, dtype=dtype))
    out: dict[str, list] = {}
    with torch.no_grad():
        model.eval()
        frames: torch.Tensor
        metadata: dict[str, torch.Tensor]
        for i, (frames, metadata) in tqdm(enumerate(demo_dataset), total=len(demo_dataset)):
            frames = frames.to(device=device, dtype=dtype).unsqueeze(0)
            # metadata = {k: v.to(device=device, dtype=dtype) for k, v in metadata.items()}
            start = time.time()
            current_phase, anticipated_phase = model(frames)
            fps = 1 / (time.time() - start)
            # print(f"FPS: {fps:.2f}")

            metadata["time_to_next_phase_dense"] = torch.clamp(metadata["time_to_next_phase_dense"], 0, model.time_horizon)
            m = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in metadata.items()}
            out_data = {
                "pred_current_phase": current_phase.cpu().float().numpy().tolist(),
                "pred_anticipated_phase": anticipated_phase.cpu().float().numpy().tolist(),
                **m,
            }
            if metadata['video_name'] not in out:
                out[metadata['video_name']] = []
            out[metadata['video_name']].append(out_data)

    with open("demo_output.pickle", "wb") as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    train_model()
