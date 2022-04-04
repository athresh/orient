import torch
# from utils.dataset_gen import DTYPE
# import tensorflow as tf

def dnse_loss(margin=1):
    def loss(y_true, y_pred):
        """Tensorflow implementation of d-SNE loss.
        Original Mxnet implementation found at https://github.com/aws-samples/d-SNE.
        @param y_true: tuple or array of two elements, containing source and target features
        @param y_pred: tuple or array of two elements, containing source and taget labels
        """
        xs = y_pred[:, 0]
        xt = y_pred[:, 1]
        ys = torch.argmax(y_true[:, 0].type(torch.int32), dim=1)
        yt = torch.argmax(y_true[:, 1].type(torch.int32), dim=1)

        batch_size = ys.shape[0]
        # tf.shape(ys)[0]
        embed_size = xs.shape[1]
        # tf.shape(xs)[1]

        # The original implementation provided an optional feature-normalisation (L2) here. We'll skip it

        xs_rpt = torch.broadcast_to(
            torch.unsqueeze(xs, dim=0), shape=(batch_size, batch_size, embed_size)
        )
        xt_rpt = torch.broadcast_to(
            torch.unsqueeze(xt, dim=1), shape=(batch_size, batch_size, embed_size)
        )

        dists = torch.sum(torch.square(xt_rpt - xs_rpt), dim=2)

        yt_rpt = torch.broadcast_to(
            torch.unsqueeze(yt, dim=1), shape=(batch_size, batch_size)
        )
        ys_rpt = torch.broadcast_to(
            torch.unsqueeze(ys, dim=0), shape=(batch_size, batch_size)
        )

        y_same = torch.eq(yt_rpt, ys_rpt)
        y_diff = torch.ne(yt_rpt, ys_rpt)

        intra_cls_dists = torch.mul(dists, y_same)
        inter_cls_dists = torch.mul(dists, y_diff)

        max_dists = torch.max(dists, dim=1, keepdim=True)
        max_dists = torch.broadcast_to(max_dists, shape=(batch_size, batch_size))
        revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)

        max_intra_cls_dist = torch.max(intra_cls_dists, dim=1)
        min_inter_cls_dist = torch.min(revised_inter_cls_dists, dim=1)
        relu = torch.nn.ReLU()
        loss = relu(max_intra_cls_dist - min_inter_cls_dist + margin)
        return loss
    return loss