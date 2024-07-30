import torch
import torch.nn as nn

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):
        P_exp = torch.exp(P) # (B,)
        P_exp_B = torch.stack([P_exp for _ in range(P.shape[0])], dim=0) # (B, B)
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        elif mode == 'surv':
            E = (M.sum(dim=1) > 0).float()
        else:
            raise NotImplementedError
        P_exp_sum = (P_exp_B * M.float()).sum(dim=1)
        P_tmp = P_exp / (P_exp_sum+1e-6)
        loss = -torch.sum(torch.log(P_tmp.clip(1e-6, P_tmp.max().item())) * E) / torch.sum(E)
        return loss

    def forward(self, P_risk, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) < T.unsqueeze(dim=0) # (B, B)
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        return loss_risk