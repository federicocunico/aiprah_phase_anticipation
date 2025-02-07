from matplotlib import pyplot as plt

# import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import pytorch_lightning as pl


class PhaseAnticipationTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, loss_criterion: torch.nn.Module):
        super().__init__()
        self.model: torch.nn.Module = model
        self.loss_criterion: torch.nn.Module = loss_criterion

        # self.ax = plt.subplot(111)  # Create a subplot for timings
        fig, self.ax = plt.subplots(1, 1, figsize=(15, 5))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        current_phase: torch.Tensor
        anticipated_phase: torch.Tensor
        loss: torch.Tensor

        frames, metadata = batch
        labels = metadata["phase_label"]
        time_to_next_phase = metadata["time_to_next_phase"]
        time_to_next_phase = torch.clamp(time_to_next_phase, 0, self.model.time_horizon)

        outputs = self.model(frames)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            current_phase, anticipated_phase = outputs
            loss, loss_phase, loss_anticipation = self.loss_criterion(
                current_phase, anticipated_phase, labels, time_to_next_phase
            )
        elif isinstance(outputs, torch.Tensor):
            regression_logits = outputs
            regression_logits = regression_logits / self.model.time_horizon  # Normalize to [0, 1]
            time_to_next_phase = time_to_next_phase / self.model.time_horizon  # Normalize to [0, 1]
            loss = self.loss_criterion(regression_logits, time_to_next_phase)
            loss_phase = 0
            loss_anticipation = loss
        else:
            current_logits, future_logits, regression_logits = outputs

            current_targets = metadata["phase_label"]  # [B, C]
            future_targets = metadata["future_targets"]  # [B, T, F, C]
            regression_targets = metadata["time_to_next_phase"]  # [B, C]
            loss_dict = self.loss_criterion(
                current_logits, future_logits, regression_logits, current_targets, future_targets, regression_targets
            )
            loss = loss_dict["total_loss"]
            loss_phase = loss_dict["classification_loss"]
            loss_anticipation = loss_dict["regression_loss"]

        self.log("train_loss", loss)
        self.log("train_loss_phase", loss_phase)
        self.log("train_loss_anticipation", loss_anticipation)
        return loss

    def on_validation_epoch_start(self):
        self.val_y_true = []
        self.val_y_pred = []
        self.val_y_true_regr = []
        self.val_y_pred_regr = []

    def on_validation_epoch_end(self):

        acc = accuracy_score(self.val_y_true, self.val_y_pred)
        mse_loss = torch.nn.functional.mse_loss(
            torch.asarray(self.val_y_pred_regr), torch.as_tensor(self.val_y_true_regr)
        )
        self.log("val_acc", acc)
        self.log("val_anticip", mse_loss)

        self.ax.cla()
        self.ax.set_ylabel("Time to next phase (cap 5)")
        self.ax.set_xlabel("Frames")
        self.ax.plot(self.val_y_true_regr, label="True", color="blue")
        self.ax.plot(self.val_y_pred_regr, label="Predicted", color="red")
        plt.legend()
        plt.savefig("val_time_phase.png")

        self.val_y_pred.clear()
        self.val_y_true.clear()
        self.val_y_pred_regr.clear()
        self.val_y_true_regr.clear()

    def validation_step(self, val_batch, batch_idx):

        frames, metadata = val_batch
        labels = metadata["phase_label"]
        time_to_next_phase = metadata["time_to_next_phase"]
        time_to_next_phase = torch.clamp(time_to_next_phase, 0, self.model.time_horizon)

        outputs = self.model(frames)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            current_phase, anticipated_phase = outputs
            loss, loss_phase, loss_anticipation = self.loss_criterion(
                current_phase, anticipated_phase, labels, time_to_next_phase
            )
            _, predicted = torch.max(current_phase, 1)
        elif isinstance(outputs, torch.Tensor):
            regression_logits = outputs
            regression_logits = regression_logits / self.model.time_horizon  # Normalize to [0, 1]
            time_to_next_phase = time_to_next_phase / self.model.time_horizon  # Normalize to [0, 1]
            loss = self.loss_criterion(regression_logits, time_to_next_phase)
            loss_phase = 0
            loss_anticipation = loss
            predicted = torch.argmin(regression_logits, dim=1)
        else:
            current_phase, future_logits, regression_logits = outputs

            current_targets = metadata["phase_label"]  # [B, C]
            future_targets = metadata["future_targets"]  # [B, T, F, C]
            regression_targets = metadata["time_to_next_phase"]  # [B, C]
            loss_dict = self.loss_criterion(
                current_phase, future_logits, regression_logits, current_targets, future_targets, regression_targets
            )
            loss = loss_dict["total_loss"]
            loss_phase = loss_dict["classification_loss"]
            loss_anticipation = loss_dict["regression_loss"]
            _, predicted = torch.max(current_phase, 1)

        self.val_y_true.extend(labels.detach().cpu().tolist())
        self.val_y_pred.extend(predicted.detach().cpu().tolist())

        # self.val_y_true_regr.extend(time_to_next_phase.detach().cpu().tolist())
        # self.val_y_pred_regr.extend(anticipated_phase.detach().cpu().tolist())

        self.log("val_loss", loss)
        self.log("val_loss_phase", loss_phase)
        self.log("val_loss_anticipation", loss_anticipation)
        return loss

    def on_test_epoch_start(self):
        self.test_y_true = []
        self.test_y_pred = []
        self.test_y_true_regr = []
        self.test_y_pred_regr = []

    def on_test_epoch_end(self):
        acc = accuracy_score(self.test_y_true, self.test_y_pred)
        f1 = f1_score(self.test_y_true, self.test_y_pred, average="weighted")
        mse_loss = torch.nn.functional.mse_loss(
            torch.as_tensor(self.test_y_pred_regr), torch.as_tensor(self.test_y_true_regr)
        )
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        self.log("test_anticip", mse_loss)

        self.ax.cla()
        self.ax.set_ylabel("Time to next phase (cap 5)")
        self.ax.set_xlabel("Frames")
        self.ax.plot(self.test_y_true_regr, label="True", color="blue")
        self.ax.plot(self.test_y_pred_regr, label="Predicted", color="red")
        plt.legend()
        plt.savefig("test_time_phase.png")

        self.test_y_true.clear()
        self.test_y_pred.clear()
        self.test_y_true_regr.clear()
        self.test_y_pred_regr.clear()

    def test_step(self, test_batch, batch_idx):
        frames, metadata = test_batch
        labels = metadata["phase_label"]
        time_to_next_phase = metadata["time_to_next_phase"]
        time_to_next_phase = torch.clamp(time_to_next_phase, 0, self.model.time_horizon)

        outputs = self.model(frames)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            current_phase, anticipated_phase = outputs
            loss, loss_phase, loss_anticipation = self.loss_criterion(
                current_phase, anticipated_phase, labels, time_to_next_phase
            )
        else:
            current_phase, future_logits, regression_logits = outputs

            current_targets = metadata["phase_label"]  # [B, C]
            future_targets = metadata["future_targets"]  # [B, T, F, C]
            regression_targets = metadata["time_to_next_phase"]  # [B, C]
            loss_dict = self.loss_criterion(
                current_phase, future_logits, regression_logits, current_targets, future_targets, regression_targets
            )
            loss = loss_dict["total_loss"]
            loss_phase = loss_dict["classification_loss"]
            loss_anticipation = loss_dict["regression_loss"]

        _, predicted = torch.max(current_phase, 1)
        self.test_y_true.extend(labels.detach().cpu().tolist())
        self.test_y_pred.extend(predicted.detach().cpu().tolist())

        # self.test_y_true_regr.extend(time_to_next_phase.detach().cpu().tolist())
        # self.test_y_pred_regr.extend(anticipated_phase.detach().cpu().tolist())

        # loss_phase: torch.Tensor = self.criterion_phase(current_phase, labels)
        # loss_anticipation: torch.Tensor = self.criterion_anticipation(anticipated_phase.reshape(-1), time_to_next_phase)

        # loss, loss_phase, loss_anticipation = self.loss_criterion(
        #     current_phase, anticipated_phase, labels, time_to_next_phase
        # )

        self.log("test_loss", loss)
        self.log("test_loss_phase", loss_phase)
        self.log("test_loss_anticipation", loss_anticipation)
        return loss
