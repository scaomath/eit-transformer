import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class SoftDiceLoss(nn.Module):
    '''
    PyTorch issue #1249
    '''
    def __init__(self, 
                 num_classes=2, 
                 smooth=1, 
                 logits=True # if true, then a sigmoid like function  is needed for post-processing
                 ):
        """:arg smooth The parameter added to the denominator and numerator to prevent the division by zero"""

        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.logits = logits

    def forward(self, preds, target, *args, **kwargs):
        assert torch.max(target).item() <= 1, 'SoftDiceLoss() is only available for binary classification.'

        batch_size = preds.size(0)

        if self.logits:
            probability = 0.5*(torch.tanh(preds)+1)
        else:
            probability = preds

        # Convert probability and target to the shape [B, (C*H*W)]
        probability = probability.view(batch_size, -1)

        if self.num_classes > 2:
            target = target.to(torch.int64)
            target = F.one_hot(target, num_classes=self.num_classes).permute((0, 3, 1, 2))
        
        target = target.contiguous().view(batch_size, -1)
        intersection = probability * target

        dsc = (2 * intersection.sum(dim=1) + self.smooth) / (
                target.sum(dim=1) + probability.sum(dim=1) + self.smooth)
        loss = dsc.mean()

        return loss


class CrossEntropyLoss2d(_WeightedLoss):
    def __init__(self,
                 regularizer=False,
                 h=1/201,  # mesh size
                 beta=1.0,
                 gamma=1e-1,  # \|N(u) - u\|_{L^1},
                 metric_reduction='mean',
                 noise=0.0,
                 label_smoothing=0.0,
                 eps=1e-8,
                 debug=False
                 ):
        super(CrossEntropyLoss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        self.h = h
        self.beta = beta 
        self.gamma = gamma  # L1
        self.label_smoothing = label_smoothing
        self.metric_reduction = metric_reduction
        self.eps = eps
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise < 1
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, preds, targets, weights=None):
        r'''
        preds: (N, n, n, 1), logits, no need to do sigmoid
        targets: (N, n, n, 1)
        loss = beta * cross entropy  + \gamma * \|N(u) - u\|^{L1}
        '''
        batch_size = targets.size(0)

        h = self.h

        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        if self.label_smoothing > 0:
            targets = self._smooth(targets, targets.size(-1),
                                   self.label_smoothing)

        inv_L2_norm = 1/torch.sqrt(targets.sum(dim=(1, 2, 3)) + self.eps)
        inv_L2_norm /= inv_L2_norm.mean()
        weights = inv_L2_norm[:, None, None,
                              None] if weights is None else weights

        loss = F.binary_cross_entropy_with_logits(preds, targets,
                                                  weight=weights,
                                                  reduction=self.metric_reduction)

        if self.regularizer:
            preds = 0.5*(torch.tanh(preds)+1) # post-processing logits
            L1_diff = ((targets - preds)).abs().mean(dim=(1, 2, 3))
            L1_diff *= weights.squeeze()
            regularizer = self.gamma * L1_diff.mean()
        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        return loss + regularizer


class L2Loss2d(_WeightedLoss):
    def __init__(self,
                 regularizer=False,
                 h=1/201,  
                 beta=0.5,  
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 delta=0.0,  #
                 metric_reduction='L1',
                 noise=0.0,
                 eps=1e-3,
                 postprocessing=True,
                 weighted=False,
                 debug=False
                 ):
        super(L2Loss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.alpha = alpha  # H^1
        self.delta = delta*h**2  # orthogonalizer
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.postprocessing = postprocessing
        self.weighted = weighted
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def forward(self, preds, targets, weights=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        '''
        batch_size = targets.size(0)

        if self.postprocessing:
            preds = 0.5*(torch.tanh(preds)+1) # postprocessing logits
            preds = 10*preds + (1-preds)
            targets = 10*targets + (1-targets)

        h = self.h if weights is None else weights
        if self.noise > 0:
            targets = L2Loss2d._noise(targets, targets.size(-1), self.noise)

        target_norm = targets.pow(2).sum(dim=(1, 2, 3)) + self.eps

        if weights is None and self.weighted:
            inv_L2_norm = 1/target_norm.sqrt()
            weights = inv_L2_norm/inv_L2_norm.mean()
        elif not self.weighted:
            weights = 1

        loss = weights*((self.beta*(preds - targets)).pow(2)).sum(dim=(1, 2, 3))/target_norm

        if self.metric_reduction == 'L2':
            loss = loss.mean().sqrt()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            loss = loss.sqrt().mean()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch to approx negative norm
            loss = loss.sqrt().max()

        if self.regularizer:
            L1_diff = ((targets - preds)).abs().mean(dim=(1, 2, 3))
            L1_diff *= weights.squeeze()
            regularizer = self.gamma * L1_diff.mean()
        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        return loss+regularizer

if __name__ == '__main__':
    
    DL = SoftDiceLoss()
    
    pred = torch.randn(2, 1, 128, 128)
    target = torch.zeros((2, 1, 128, 128)).long()

    dl_loss = DL(pred, target)

    print('testing dice loss:', dl_loss.item())