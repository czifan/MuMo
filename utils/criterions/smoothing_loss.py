from torch import nn
from torch.nn import functional as F

# adapted from Fairseq

class CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, ls_eps=0.1, ignore_idx=None, reduce=True, reduction='mean', *args, **kwargs):
        super(CrossEntropyWithLabelSmoothing, self).__init__()
        self.ls_eps = ls_eps
        self.ignore_idx = ignore_idx
        self.reduce = reduce
        self.reduction = reduction

    def compute_loss(self, log_probs, target):
        if target.dim() == log_probs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -log_probs.gather(dim=-1, index=target)
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)
        if self.ignore_idx is not None:
            pad_mask = target.eq(self.ignore_idx)
            if pad_mask.any():
                nll_loss.masked_fill_(pad_mask, 0.)
                smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.ls_eps / log_probs.size(-1)
        loss = (1. - self.ls_eps) * nll_loss + eps_i * smooth_loss
        return loss

    def forward(self, pred, target):
        assert pred.dim() == 2, 'Should be B x C'
        B, C = pred.size()
        log_probs = F.log_softmax(pred, dim=-1)
        log_probs = log_probs.view(-1, C)
        target = target.view(-1, 1)
        loss = self.compute_loss(log_probs, target)
        if self.reduction == 'mean':
            loss /= B
        return loss
