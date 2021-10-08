from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SSL import GradMatchStrategy
import time, copy, torch


class GradMatchDataLoader(AdaptiveDSSDataLoader):

    def __init__(self, train_loader, val_loader, dss_args, verbose=True, *args, **kwargs):
        
        """
         Arguments assertion check
        """
        assert "model" in dss_args.keys(), "'model' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "tea_model" in dss_args.keys(), "'tea_model' is a compulsory argument. Include it as a key in dss_args"
        assert "ssl_alg" in dss_args.keys(), "'ssl_alg' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduce:
            raise ValueError("Please set 'reduce' of loss function to False for adaptive subset selection strategies")
        assert "eta" in dss_args.keys(), "'eta' is a compulsory argument. Include it as a key in dss_args"
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "valid" in dss_args.keys(), "'valid' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "v1" in dss_args.keys(), "'v1' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "lam" in dss_args.keys(), "'lam' is a compulsory argument for GradMatch. Include it as a key in dss_args"
        assert "eps" in dss_args.keys(), "'eps' is a compulsory argument for GradMatch. Include it as a key in dss_args"

        super(GradMatchDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                     verbose=verbose, *args, **kwargs)
        self.strategy = GradMatchStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), copy.deepcopy(dss_args.tea_model),
                                         dss_args.ssl_alg, dss_args.loss, dss_args.eta, dss_args.device, dss_args.num_classes, 
                                         dss_args.linear_layer, dss_args.selection_type, dss_args.valid, dss_args.v1,
                                         dss_args.lam, dss_args.eps)
        self.train_model = dss_args.model
        self.teacher_model = dss_args.tea_model
        if self.verbose:
            print('Grad-match dataloader initialized. ')

    def _resample_subset_indices(self):
        if self.verbose:
            start = time.time()
            print('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        tea_cached_state_dict = copy.deepcopy(self.teacher_model.state_dict())
        tea_clone_dict = copy.deepcopy(self.teacher_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict, tea_clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        self.teacher_model.load_state_dict(tea_cached_state_dict)
        if self.verbose:
            end = time.time()
            print('Epoch: {0:d}, subset selection finished, takes {1:.2f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
