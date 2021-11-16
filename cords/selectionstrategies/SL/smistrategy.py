import math
import numpy as np
import time
import torch
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
import submodlib
# from submodlib import FacilityLocationMutualInformationFunction, FacilityLocationVariantMutualInformationFunction

class SMIStrategy(DataSelectionStrategy):
    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer,
                 selection_type, logger, smi_func_type, valid=True, optimizer='NaiveGreedy', metric='cosine', eta=1,
                 stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.selection_type = selection_type
        self.logger = logger
        self.optimizer = optimizer
        self.smi_func_type = smi_func_type
        self.valid = valid
        self.metric = metric
        self.eta = eta
        self.stopIfZeroGain = stopIfZeroGain
        self.stopIfNegativeGain = stopIfNegativeGain
        self.verbose = verbose

    def select(self, budget, model_params):
        """

        Parameters
        ----------
        budget :
        model_params :

        Returns
        -------

        """
        # start_time = time.time()
        # for batch_idx, (inputs, targets) in enumerate(self.trainloader):
        #     if batch_idx == 0:
        #         labels = targets
        #     else:
        #         tmp_target_i = targets
        #         labels = torch.cat((labels, tmp_target_i), dim=0)
        # total_greedy_list = []
        # gammas = []
        # if self.selection_type == 'PerBatch':
        #     for i in range(self.num_classes):
        #         if i == 0:
        #             idxs = torch.where(labels == i)[0]
        #             N = len(idxs)
        #             self.compute_score(model_params, idxs)
        #             row = idxs.repeat_interleave(N)
        #             col = idxs.repeat(N)
        #             data = self.dist_mat.flatten()
        #         else:
        #             idxs = torch.where(labels == i)[0]
        #             N = len(idxs)
        #             self.compute_score(model_params, idxs)
        #             row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
        #             col = torch.cat((col, idxs.repeat(N)), dim=0)
        #             data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
        #     sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
        #     self.dist_mat = sparse_simmat
        #     fl = FacilityLocationMutualInformationFunction()
        smi_start_time = time.time()
        if self.selection_type == 'Supervised':
            self.compute_gradients(self.valid)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            val_gradients = self.val_grads_per_elem
            # if self.valid:
            #     sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            # else:
            #     sum_val_grad = torch.sum(trn_gradients, dim=0)

            data_sijs = submodlib.helper.create_kernel(X=trn_gradients.cpu().numpy(), metric=self.metric, method='sklearn')
            query_sijs = submodlib.helper.create_kernel(X=val_gradients.cpu().numpy(), X_rep=trn_gradients.cpu().numpy(), metric=self.metric,
                                                         method='sklearn')

            if self.smi_func_type == 'fl1mi':
                obj = submodlib.FacilityLocationMutualInformationFunction(n=self.N_trn,
                                                                num_queries=self.N_val,
                                                                data_sijs=data_sijs,
                                                                query_sijs=query_sijs,
                                                                magnificationEta=self.eta)
            if self.smi_func_type == 'fl2mi':
                obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=self.N_trn,
                                                                num_queries=self.N_val,
                                                                query_sijs=query_sijs,
                                                                queryDiversityEta=self.eta)
            greedyList = obj.maximize(budget=budget, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                      stopIfNegativeGain=self.stopIfNegativeGain, verbose=self.verbose)
            greedyIdxs = [x[0] for x in greedyList]
            gammas = [1]*budget
            smi_end_time = time.time()
            self.logger.debug("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return greedyIdxs, gammas
