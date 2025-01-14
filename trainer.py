from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import pytorch_lightning as pl



class PhaseAnticipationTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, loss_criterion: torch.nn.Module):
        super().__init__()
        self.model: torch.nn.Module = model
        self.loss_criterion: torch.nn.Module = loss_criterion

        self.ax = plt.subplot(111)  # Create a subplot for timings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        current_phase: torch.Tensor
        anticipated_phase: torch.Tensor
        loss: torch.Tensor

        frames, metadata = batch
        labels = metadata["phase_label"]
        time_to_next_phase = metadata["time_to_next_phase"]
        time_to_next_phase = torch.clamp(time_to_next_phase, 0, self.model.time_horizon)

        current_phase, anticipated_phase = self.model(frames)

        loss, loss_phase, loss_anticipation = self.loss_criterion(current_phase, anticipated_phase, labels, time_to_next_phase)

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
            torch.as_tensor(self.val_y_pred_regr), torch.as_tensor(self.val_y_true_regr)
        )
        self.log("val_acc", acc)
        self.log("val_mse_loss", mse_loss)

        self.ax.cla()
        self.ax.set_ylabel("Time to next phase (cap 5)")
        self.ax.set_xlabel("Frames")
        self.ax.plot(np.asarray(self.val_y_true_regr), label="True", color="blue")
        self.ax.plot(np.asarray(self.val_y_pred_regr), label="Predicted", color="red")
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

        current_phase, anticipated_phase = self.model(frames)

        _, predicted = torch.max(current_phase, 1)
        self.val_y_true.extend(labels.detach().cpu())
        self.val_y_pred.extend(predicted.detach().cpu())

        self.val_y_true_regr.extend(time_to_next_phase.detach().cpu())
        self.val_y_pred_regr.extend(anticipated_phase.detach().cpu())

        
        loss, loss_phase, loss_anticipation = self.loss_criterion(current_phase, anticipated_phase, labels, time_to_next_phase)

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
        mse_loss = self.criterion_anticipation(
            torch.as_tensor(self.test_y_pred_regr), torch.as_tensor(self.test_y_true_regr)
        )
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        self.log("test_mse_loss", mse_loss)

        self.ax.cla()
        self.ax.set_ylabel("Time to next phase (cap 5)")
        self.ax.set_xlabel("Frames")
        self.ax.plot(np.asarray(self.test_y_true_regr), label="True", color="blue")
        self.ax.plot(np.asarray(self.test_y_pred_regr), label="Predicted", color="red")
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

        current_phase, anticipated_phase = self.model(frames)

        _, predicted = torch.max(current_phase, 1)
        self.test_y_true.extend(labels.detach().cpu())
        self.test_y_pred.extend(predicted.detach().cpu())

        self.test_y_true_regr.extend(time_to_next_phase.detach().cpu())
        self.test_y_pred_regr.extend(anticipated_phase.detach().cpu())

        # loss_phase: torch.Tensor = self.criterion_phase(current_phase, labels)
        # loss_anticipation: torch.Tensor = self.criterion_anticipation(anticipated_phase.reshape(-1), time_to_next_phase)

        loss, loss_phase, loss_anticipation = self.loss_criterion(current_phase, anticipated_phase, labels, time_to_next_phase)

        self.log("test_loss", loss)
        self.log("test_loss_phase", loss_phase)
        self.log("test_loss_anticipation", loss_anticipation)
        return loss
