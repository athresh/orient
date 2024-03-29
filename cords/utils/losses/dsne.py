import torch
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import CrossEntropyLoss


class DSNELoss(_Loss):
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
        2. Find the minimum interclass distance (y_trn != y_cmp) and
            maximum intraclass distance (y_trn == y_cmp)
        3. Check that the difference between these two distances is
            greater than a specified margin.
        Explanation:
            -The minimum interclass distance should be large, as FVs
                from src/tgt pairs should be as distinct as possible
                when their classes are different.
            -The maximum intraclass distance should be small, as FVs
                from src/tgt pairs should be as similar as possible
                when their classes are the same.
            -Therefore, these conditions should be true:
                              `min_interclass` >> `max_interclass`
           `min_interclass` - `max_interclass` >> `margin`
        4. Calculate loss for cases where the difference is NOT greater
            than the margin, as that would invalidate the conditions
            above. Here, loss == abs(difference).
    """

    def __init__(self, margin=1.0, reduction="mean"):
        """Assign parameters as attributes."""
        super(DSNELoss, self).__init__()

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
        dists = torch.sum((ft_src_rpt - ft_tgt_rpt)**2, dim=2)

        # Split <source, target> distances into 2 groups:
        #   1. intraclass distances (y_src == y_tgt)
        #   2. interclass distances (y_src != y_tgt)
        y_src_rpt = y_src.unsqueeze(1).expand(broadcast_size[0:2])
        y_tgt_rpt = y_tgt.unsqueeze(0).expand(broadcast_size[0:2])
        y_same = torch.eq(y_src_rpt, y_tgt_rpt)   # Boolean mask
        y_diff = torch.logical_not(y_same)        # Boolean mask
        intraclass_dists = dists * y_same   # Set 0 where classes are different
        interclass_dists = dists * y_diff   # Set 0 where classes are the same

        # Fill 0 values with max to prevent interference with min calculation
        max_dists = torch.max(dists, dim=1, keepdim=True)[0]
        max_dists = max_dists.expand(broadcast_size[0:2])
        interclass_dists = torch.where(y_same, max_dists, interclass_dists)

        # For each training image, find the minimum interclass distance
        min_interclass_dist = interclass_dists.min(1)[0]

        # For each training image, find the maximum intraclass distance
        max_intraclass_dist = intraclass_dists.max(1)[0]

        # No loss for differences greater than margin (clamp to 0)
        differences = min_interclass_dist.sub(max_intraclass_dist)
        loss = torch.abs(differences.sub(self.margin).clamp(max=0))
        if self.reduce_func is not None:
            loss = self.reduce_func(loss)
        return loss