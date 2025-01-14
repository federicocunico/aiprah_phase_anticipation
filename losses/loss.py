import torch


class ModelLoss(torch.nn.Module):
    def __init__(self, l1: float, l2: float, multitask_strategy: str | None):
        super(ModelLoss, self).__init__()
        self.multitask_strategy = multitask_strategy
        self.l1 = l1
        self.l2 = l2

        self.criterion_phase = torch.nn.CrossEntropyLoss()
        self.criterion_anticipation = torch.nn.MSELoss()

    def forward(
        self,
        current_phase: torch.Tensor,
        anticipated_phase: torch.Tensor,
        labels: torch.Tensor,
        time_to_next_phase: torch.Tensor,
    ):
        loss_phase: torch.Tensor = self.criterion_phase(current_phase, labels)
        loss_anticipation: torch.Tensor = self.criterion_anticipation(anticipated_phase.reshape(-1), time_to_next_phase)

        if self.multitask_strategy == "lambda":
            loss = self.l1 * loss_phase + self.l2 * loss_anticipation
        elif self.multitask_strategy == "none" or self.multitask_strategy is None:
            loss = loss_phase + loss_anticipation
        else:
            raise ValueError(f"Unknown multitask strategy: {self.multitask_strategy}")

        return loss, loss_phase, loss_anticipation
