from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch
import pytorch_lightning as pl


class PhaseAnticipationTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, loss_criterion: torch.nn.Module, multi_task: bool = False):
        super().__init__()
        self.model: torch.nn.Module = model
        self.loss_criterion: torch.nn.Module = loss_criterion
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.multi_task = multi_task
        _, self.ax = plt.subplots(1, 1, figsize=(15, 5))
        self.min_loss = float("inf")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)

    # ----------------- helpers -----------------
    @staticmethod
    def _parse_outputs(outputs):
        """
        Normalize model outputs to a dict:
          - If tuple length == 2: (current_phase, anticipated_phase)
          - If tensor only: regression head only
          - If tuple length >= 3: assume current at [0], regression at [-1], future at [1] (optional)
        """
        out = {"current": None, "future": None, "reg": None, "anticipated": None, "raw": outputs}
        if isinstance(outputs, torch.Tensor):
            out["reg"] = outputs
            return out
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 2:
                out["current"], out["anticipated"] = outputs
                return out
            # 3 or more: be tolerant
            out["current"] = outputs[0]
            out["reg"] = outputs[-1]
            if len(outputs) >= 3:
                out["future"] = outputs[1]
            return out
        raise TypeError(f"Unsupported outputs type: {type(outputs)}")

    def _pred_times_same_unit(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        """
        Convert model outputs into SAME unit (seconds/minutes) as dataset targets. Returns [B, C].
        """
        parsed = self._parse_outputs(outputs)
        if parsed["anticipated"] is not None:
            pred = torch.clamp(parsed["anticipated"], 0.0, self.model.time_horizon)
            return pred
        if parsed["reg"] is not None:
            # if you trained the regression head with normalized targets (0..1), scale back here
            # NOTE: in training/val/test steps we normalize before loss where needed;
            # here we always bring predictions back to real units for metrics.
            pred = parsed["reg"]
            # If your regression head already outputs real units, this next line is a no-op if values are ≤ horizon.
            # If it outputs [0..1], comment in the scaling below OR keep it clamped if already scaled upstream.
            # pred = pred * self.model.time_horizon
            pred = torch.clamp(pred, 0.0, self.model.time_horizon)
            return pred
        raise RuntimeError("No regression/anticipated output found to compute time predictions.")

    @staticmethod
    def _batch_mae(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - tgt))

    def get_loss(self, phase_loss, anticipation_loss):
        if self.multi_task:
            return 0.01 * phase_loss + 1 * anticipation_loss
        return anticipation_loss

    # ----------------- training -----------------
    def training_step(self, batch, batch_idx):
        frames, metadata = batch
        labels = metadata["phase_label"]
        time_to_next_phase = torch.clamp(metadata["time_to_next_phase"], 0, self.model.time_horizon)  # [B, C]

        outputs: torch.Tensor = self.model(frames)  # [B, C]
        predicted_class = outputs.argmin(dim=1).float()

        phase_loss = self.cross_entropy_loss(predicted_class, labels)
        anticipation_loss = self.loss_criterion(outputs, time_to_next_phase)

        train_loss = self.get_loss(phase_loss, anticipation_loss)

        with torch.no_grad():
            train_mae = self._batch_mae(outputs, time_to_next_phase)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_loss_anticipation", anticipation_loss)
        self.log("phase_loss", phase_loss)
        self.log("train_mae", train_mae, prog_bar=True)
        return train_loss

    # ----------------- validation -----------------
    def on_validation_epoch_start(self):
        self.val_y_true, self.val_y_pred = [], []
        self.val_y_true_regr, self.val_y_pred_regr = [], []

    def validation_step(self, val_batch, batch_idx):
        frames, metadata = val_batch
        labels = metadata["phase_label"]
        time_to_next_phase = torch.clamp(metadata["time_to_next_phase"], 0, self.model.time_horizon)

        outputs: torch.Tensor = self.model(frames)
        predicted_class = outputs.argmin(dim=1).float()

        phase_loss = self.cross_entropy_loss(predicted_class, labels)
        anticipation_loss = self.loss_criterion(outputs, time_to_next_phase)

        val_loss = self.get_loss(phase_loss, anticipation_loss)

        self.val_y_true.extend(labels.detach().cpu().tolist())
        self.val_y_pred.extend(predicted_class.detach().cpu().tolist())

        with torch.no_grad():
            val_mae = self._batch_mae(outputs, time_to_next_phase)
            # store per-frame averages (mean over classes) for plotting
            self.val_y_true_regr.extend(time_to_next_phase.mean(dim=1).detach().cpu().tolist())
            self.val_y_pred_regr.extend(outputs.mean(dim=1).detach().cpu().tolist())

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_loss_anticipation", anticipation_loss)
        self.log("val_loss_phase", phase_loss)
        self.log("val_mae", val_mae, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        acc = accuracy_score(self.val_y_true, self.val_y_pred)
        mse_loss = torch.nn.functional.mse_loss(
            torch.tensor(self.val_y_pred_regr), torch.tensor(self.val_y_true_regr)
        )
        self.log("val_acc", acc)
        self.log("val_anticip_mse", mse_loss)

        if mse_loss < self.min_loss:
            self.min_loss = mse_loss
            torch.save(self.model.state_dict(), "best.pth")

        self.ax.cla()
        self.ax.set_ylabel(f"Time to next phase (cap {self.model.time_horizon})")
        self.ax.set_xlabel("Frames")
        self.ax.plot(self.val_y_true_regr, label="True", color="blue")
        self.ax.plot(self.val_y_pred_regr, label="Predicted", color="red")
        plt.legend()
        plt.savefig("val_time_phase.png")

        self.val_y_pred.clear(); self.val_y_true.clear()
        self.val_y_pred_regr.clear(); self.val_y_true_regr.clear()

    # ----------------- test -----------------
    def on_test_epoch_start(self):
        self.test_y_true, self.test_y_pred = [], []
        self.test_y_true_regr, self.test_y_pred_regr = [], []

    def test_step(self, test_batch, batch_idx):
        frames, metadata = test_batch
        labels = metadata["phase_label"]
        time_to_next_phase = torch.clamp(metadata["time_to_next_phase"], 0, self.model.time_horizon)

        outputs: torch.Tensor = self.model(frames)
        predicted_class = outputs.argmin(dim=1).float()

        phase_loss = self.cross_entropy_loss(predicted_class, labels)
        anticipation_loss = self.loss_criterion(outputs, time_to_next_phase)

        test_loss = self.get_loss(phase_loss, anticipation_loss)

        self.test_y_true.extend(labels.detach().cpu().tolist())
        self.test_y_pred.extend(predicted_class.detach().cpu().tolist())

        with torch.no_grad():
            test_mae = self._batch_mae(outputs, time_to_next_phase)
            self.test_y_true_regr.extend(time_to_next_phase.mean(dim=1).detach().cpu().tolist())
            self.test_y_pred_regr.extend(outputs.mean(dim=1).detach().cpu().tolist())

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_loss_anticipation", anticipation_loss)
        self.log("test_loss_phase", phase_loss)
        self.log("test_mae", test_mae, prog_bar=True)
        return test_loss

    def on_test_epoch_end(self):
        acc = accuracy_score(self.test_y_true, self.test_y_pred)
        f1 = f1_score(self.test_y_true, self.test_y_pred, average="weighted")
        mse_loss = torch.nn.functional.mse_loss(
            torch.tensor(self.test_y_pred_regr), torch.tensor(self.test_y_true_regr)
        )
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        self.log("test_anticip_mse", mse_loss)

        self.ax.cla()
        self.ax.set_ylabel(f"Time to next phase (cap {self.model.time_horizon})")
        self.ax.set_xlabel("Frames")
        self.ax.plot(self.test_y_true_regr, label="True", color="blue")
        self.ax.plot(self.test_y_pred_regr, label="Predicted", color="red")
        plt.legend()
        plt.savefig("test_time_phase.png")

        self.test_y_true.clear(); self.test_y_pred.clear()
        self.test_y_true_regr.clear(); self.test_y_pred_regr.clear()
