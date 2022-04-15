import torch
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

def euclidean_distance(x1, x2):
    return torch.sqrt(torch.maximum(torch.sum(torch.square(x1 - x2), dim=1, keepdim=False), 1e-08))


def labels_equal(y1, y2):
    return torch.all(torch.eq(y1, y2), dim=1, keepdim=False)


class CCSALoss(_Loss):
    """d-SNE loss for paired batches of source and target features.
    Attributes
    ----------
    margin : float
        Minimum required margin between `min_intraclass_dist` and
        `max_interclass_dist`.
    reduction : str
        Name of torch function for reducing loss vector to a scalar. Set
        to "mean" by default to mimic CrossEntropyLoss default
        parameter.
    Notes
    -----
    The loss calculation involves the following plain-English steps:
    For each image in the training batch...
        1. Compare its feature vector to all FVs in the comparison batch
            using a distance function. (L2-norm/Euclidean used here.)
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """Assign parameters as attributes."""
        super(CCSALoss, self).__init__()
        self.margin = margin
        if reduction == "mean":
            self.reduce_func = torch.mean
        elif reduction == "none":
            self.reduce_func = None
        else:
            raise NotImplementedError

    def forward(self, ft_src, ft_tgt, y_src, y_tgt):
        """Compute forward pass for loss function.
        Parameters
        ----------
        ft_src : dict of PyTorch Tensors (N, F)
            Source feature embedding vectors.
        ft_tgt : dict of PyTorch Tensors (M, F)
            Target feature embedding vectors.
        y_src : dict of PyTorch Tensors (N)
            Source labels.
        y_tgt : dict of PyTorch Tensors (M)
            Target labels.
        """
        dist = F.pairwise_distance(ft_src, ft_tgt, p=2)
        class_eq = torch.eq(y_src, y_tgt)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (self.margin - dist).clamp(min=0).pow(2)
        if self.reduce_func is not None:
            loss = self.reduce_func(loss)
        return loss
