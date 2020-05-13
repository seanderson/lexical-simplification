import torch as tt
from torch.nn.modules.loss import _Loss


class AlignmentLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(AlignmentLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, diff, mask):
        mask_simple = mask * mask
        max_mask = (diff * mask_simple == tt.max(diff * mask_simple, dim=1)[0].unsqueeze(1).repeat((1, diff.shape[1], 1))).float()
        cases = tt.sum(max_mask * mask != 0, dim=(1, 2))
        loss = tt.mean(tt.sum(tt.nn.functional.relu(diff * max_mask * mask + 0.5) * mask_simple, dim=(1, 2)) / cases)
        acc = tt.mean(tt.sum(diff * max_mask * mask < 0, dim=(1, 2)).float() / cases)
        return loss, acc
