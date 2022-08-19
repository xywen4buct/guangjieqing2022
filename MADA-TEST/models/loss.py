import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma, alpha=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        # y_pred_int = torch.round(y_pred)
        # 输入的y_true和y_pred都为Tensor
        p_t = torch.multiply(y_true,y_pred) + torch.multiply((torch.ones_like(y_true) - y_true),
                                                             (torch.ones_like(y_pred) - y_pred))
        ce_loss = -torch.log(p_t)
        loss = torch.multiply(torch.pow((torch.ones_like(y_pred) - p_t),self.gamma),ce_loss)
        if self.alpha:
            alpha_t = self.alpha * (y_true) + (1-self.alpha) * \
                      (torch.ones_like(y_true) - y_true)
            loss = torch.multiply(alpha_t,loss)
        return torch.mean(loss)
