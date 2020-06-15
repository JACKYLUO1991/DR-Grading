from __future__ import print_function, absolute_import

import torch
import torch.nn as nn

__all__ = ['accuracy', 'kl_loss', 'feature_loss', 'FocalLoss']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def kl_loss(logits_q, logits_p, T):
    assert logits_p.size() == logits_q.size()
    b, c = logits_p.size()
    p = nn.Softmax(dim=1)(logits_p / T)
    q = nn.Softmax(dim=1)(logits_q / T)
    epsilon = 1e-8
    _p = (p + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    _q = (q + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    return (T ** 2) * torch.mean(torch.sum(_p * torch.log(_p / _q), dim=1))


# def kl_loss(output, target_output, args):
#     """Compute kd loss"""
#     """
#     para: output: middle ouptput logits.
#     para: target_output: final output has divided by temperature and softmax.
#     """
#
#     output = output / args.temperature
#     output_log_softmax = torch.log_softmax(output, dim=1)
#     loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
#     return loss_kd


def feature_loss(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean()
