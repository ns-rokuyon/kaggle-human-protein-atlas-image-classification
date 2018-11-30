import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()


def f1_loss(logits, labels):
    """Differentiable F1 loss with logits
    """
    eps = 1e-6
    beta = 1
    batch_size = logits.size()[0]
    p = F.sigmoid(logits)
    l = labels
    num_pos = torch.sum(p, 1) + eps
    num_pos_hat = torch.sum(l, 1) + eps
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + eps)
    loss = fs.sum() / batch_size
    return (1 - loss)