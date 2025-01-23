import torch


class CERegrLoss(torch.nn.Module):
    def __init__(self):
        super(CERegrLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.regr_loss = torch.nn.MSELoss()

    def forward(self, phase_pred, time_pred, phase_gt, time_gt):
        loss_phase = self.ce_loss(phase_pred, phase_gt)
        loss_time = self.regr_loss(time_pred, time_gt)
        loss = loss_phase + loss_time
        return loss, loss_phase, loss_time
