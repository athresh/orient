import torch
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import CrossEntropyLoss


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
        # If training batch -> (N, F) and comparison batch -> (M, F), then
        # distances for all combinations of pairs will be of shape (N, M, F)
        broadcast_size = (ft_src.shape[0], ft_tgt.shape[0], ft_src.shape[1])

        # Compute distances between all <train, comparison> pairs of vectors
        ft_src_rpt = ft_src.unsqueeze(1).expand(broadcast_size)
        ft_tgt_rpt = ft_tgt.unsqueeze(0).expand(broadcast_size)
        dists = 0.5 * torch.sum((ft_src_rpt - ft_tgt_rpt)**2, dim=2)

        # Split <source, target> distances into 2 groups:
        #   1. intraclass distances (y_src == y_tgt)
        #   2. interclass distances (y_src != y_tgt)
        y_src_rpt = y_src.unsqueeze(1).expand(broadcast_size[0:2])
        y_tgt_rpt = y_tgt.unsqueeze(0).expand(broadcast_size[0:2])
        y_same = torch.eq(y_src_rpt, y_tgt_rpt)   # Boolean mask
        y_diff = torch.logical_not(y_same)        # Boolean mask
        intraclass_dists = dists * y_same   # Set 0 where classes are different
        interclass_dists = torch.maximum(self.margin - (dists * y_diff), torch.tensor(0).to(dists.device))   # Set 0 where classes are the same

        # No loss for differences greater than margin (clamp to 0)
        loss = intraclass_dists.sum(dim=1) + interclass_dists.sum(dim=1)
        if self.reduce_func is not None:
            loss = self.reduce_func(loss)
        return loss
