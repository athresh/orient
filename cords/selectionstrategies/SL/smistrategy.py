import math
import numpy as np
import time
import torch
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import submodlib
# from submodlib import FacilityLocationMutualInformationFunction, FacilityLocationVariantMutualInformationFunction

class SMIStrategy(DataSelectionStrategy):
    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer,
                 selection_type, logger, smi_func_type, query_size, valid=True, optimizer='NaiveGreedy', similarity_criterion='gradient', metric='cosine', eta=1,
                 stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, lambdaVal=1):
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
        # self.query_size = len(valloader.dataset)/len(valloader)
        # self.query_size = int(np.ceil(0.01*len(valloader.dataset)))
        self.query_size = query_size
        self.verbose = verbose
        self.lambdaVal = lambdaVal
        self.similarity_criterion = similarity_criterion
    
    def compute_gradients(self, valid=False, perBatch=False, perClass=False):
        """
        Computes the gradient of each element.

        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.

        Using different loss functions, the way we calculate the gradients will change.

        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,

        where :math:`N` is the batch size.


        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,

        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        perBatch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if (perBatch and perClass):
            raise ValueError("batch and perClass are mutually exclusive. Only one of them can be true at a time")

        embDim = self.model.get_embedding_dim()
        if perClass:
            trainloader = self.pctrainloader
            if valid:
                valloader = self.pcvalloader
        else:
            trainloader = self.trainloader
            if valid:
                valloader = self.valloader
        # for batch_idx, (inputs, targets, domains) in enumerate(trainloader):
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        for batch_idx, batch in enumerate(trainloader):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, domain = batch
            else:
                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if batch_idx == 0:
                out, l1 = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                if perBatch:
                    l0_grads = l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        l1_grads = l1_grads.mean(dim=0).view(1, -1)
            else:
                out, l1 = self.model(inputs, last=True, freeze=True)
                loss = self.loss(out, targets).sum()
                batch_l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                    batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                if perBatch:
                    batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                if self.linear_layer:
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        torch.cuda.empty_cache()

        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads

        if valid:
            # for batch_idx, (inputs, targets, domains) in enumerate(valloader):
            # for batch_idx, (inputs, targets) in enumerate(valloader):
            for batch_idx, batch in enumerate(valloader):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    inputs, targets, domain = batch
                else:
                    raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    # if self.linear_layer:
                    #     self.query_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
                    # else:
                    #     self.query_grads_per_elem = l0_grads
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads

            self.query_grads_per_elem = self.val_grads_per_elem[0:self.query_size, :]

    # def get_data(self, valid):
    #     trn_data, trn_targets = self.trainloader.Dataset[:]
    #

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
            if self.similarity_criterion == "gradient":
                self.compute_gradients(self.valid)
                trn_gradients = self.grads_per_elem
                query_gradients = self.query_grads_per_elem

                query_sijs = submodlib.helper.create_kernel(X=query_gradients.cpu().numpy(),
                                                            X_rep=trn_gradients.cpu().numpy(), metric=self.metric,
                                                            method='sklearn')

                if self.smi_func_type in ['fl1mi', 'logdetmi']:
                    data_sijs = submodlib.helper.create_kernel(X=trn_gradients.cpu().numpy(), metric=self.metric,
                                                               method='sklearn')
                if self.smi_func_type in ['logdetmi']:
                    query_query_sijs = submodlib.helper.create_kernel(X=query_gradients.cpu().numpy(),
                                                                      metric=self.metric,
                                                                      method='sklearn')
            elif self.similarity_criterion == "feature":
                trn_data, trn_targets = self.trainloader.dataset[:]
                query_data, query_targets = self.valloader.dataset[0:self.query_size]
                trn_data = np.hstack((trn_data, trn_targets[:, None]))
                query_data = np.hstack((query_data, query_targets[:, None]))
                query_sijs = submodlib.helper.create_kernel(X=query_data,
                                                            X_rep=trn_data, metric=self.metric,
                                                            method='sklearn')
                if self.smi_func_type in ['fl1mi', 'logdetmi']:
                    data_sijs = submodlib.helper.create_kernel(X=trn_data, metric=self.metric,
                                                               method='sklearn')
                if self.smi_func_type in ['logdetmi']:
                    query_query_sijs = submodlib.helper.create_kernel(X=query_data,
                                                                      metric=self.metric,
                                                                      method='sklearn')

            if self.smi_func_type == 'fl1mi':
                obj = submodlib.FacilityLocationMutualInformationFunction(n=self.N_trn,
                                                                num_queries=self.query_size,
                                                                data_sijs=data_sijs,
                                                                query_sijs=query_sijs,
                                                                magnificationEta=self.eta)
            if self.smi_func_type == 'fl2mi':
                obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=self.N_trn,
                                                                num_queries=self.query_size,
                                                                query_sijs=query_sijs,
                                                                queryDiversityEta=self.eta)
            if self.smi_func_type == 'logdetmi':
                obj = submodlib.LogDeterminantMutualInformationFunction(n=self.N_trn,
                                                                        num_queries=self.query_size,
                                                                        data_sijs=data_sijs,
                                                                        lambdaVal=self.lambdaVal,
                                                                        query_sijs=query_sijs,
                                                                        query_query_sijs=query_query_sijs,
                                                                        magnificationEta=self.eta
                                                                        )
            if self.smi_func_type == 'gcmi':
                obj = submodlib.GraphCutMutualInformationFunction(n=self.N_trn,
                                                                  num_queries=self.query_size,
                                                                  query_sijs=query_sijs)
            greedyList = obj.maximize(budget=budget, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                      stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)
            greedyIdxs = [x[0] for x in greedyList]
            gammas = [1]*budget

        elif self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            greedyIdxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                query_size_per_class = np.ceil(self.query_size/self.num_classes)
                N_trn_per_class = len(trn_subset_idx)
                budget_per_class = np.ceil(budget/self.num_classes)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  collate_fn=self.trainloader.collate_fn)
                if self.similarity_criterion == "gradient":
                    self.compute_gradients(valid=self.valid, perClass=True)
                    trn_gradients = self.grads_per_elem
                    query_gradients = self.query_grads_per_elem

                    query_sijs = submodlib.helper.create_kernel(X=query_gradients.cpu().numpy(),
                                                                X_rep=trn_gradients.cpu().numpy(), metric=self.metric,
                                                                method='sklearn')

                    if self.smi_func_type in ['fl1mi', 'logdetmi']:
                        data_sijs = submodlib.helper.create_kernel(X=trn_gradients.cpu().numpy(), metric=self.metric,
                                                                   method='sklearn')
                    if self.smi_func_type in ['logdetmi']:
                        query_query_sijs = submodlib.helper.create_kernel(X=query_gradients.cpu().numpy(),
                                                                          metric=self.metric,
                                                                          method='sklearn')
                elif self.similarity_criterion == "feature":
                    trn_data, trn_targets = self.pctrainloader.dataset[:]
                    query_data, query_targets = self.pcvalloader.dataset[0:query_size_per_class]
                    trn_data = np.hstack((trn_data, trn_targets[:, None]))
                    query_data = np.hstack((query_data, query_targets[:, None]))
                    query_sijs = submodlib.helper.create_kernel(X=query_data,
                                                                X_rep=trn_data, metric=self.metric,
                                                                method='sklearn')
                    if self.smi_func_type in ['fl1mi', 'logdetmi']:
                        data_sijs = submodlib.helper.create_kernel(X=trn_data, metric=self.metric,
                                                                   method='sklearn')
                    if self.smi_func_type in ['logdetmi']:
                        query_query_sijs = submodlib.helper.create_kernel(X=query_data,
                                                                          metric=self.metric,
                                                                          method='sklearn')

                if self.smi_func_type == 'fl1mi':
                    obj = submodlib.FacilityLocationMutualInformationFunction(n=N_trn_per_class,
                                                                    num_queries=query_size_per_class,
                                                                    data_sijs=data_sijs,
                                                                    query_sijs=query_sijs,
                                                                    magnificationEta=self.eta)
                if self.smi_func_type == 'fl2mi':
                    obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=N_trn_per_class,
                                                                    num_queries=query_size_per_class,
                                                                    query_sijs=query_sijs,
                                                                    queryDiversityEta=self.eta)
                if self.smi_func_type == 'logdetmi':
                    obj = submodlib.LogDeterminantMutualInformationFunction(n=N_trn_per_class,
                                                                            num_queries=query_size_per_class,
                                                                            data_sijs=data_sijs,
                                                                            lambdaVal=self.lambdaVal,
                                                                            query_sijs=query_sijs,
                                                                            query_query_sijs=query_query_sijs,
                                                                            magnificationEta=self.eta
                                                                            )
                if self.smi_func_type == 'gcmi':
                    obj = submodlib.GraphCutMutualInformationFunction(n=N_trn_per_class,
                                                                      num_queries=query_size_per_class,
                                                                      query_sijs=query_sijs)
                greedyListPerClass = obj.maximize(budget=budget, optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain,
                                          stopIfNegativeGain=self.stopIfNegativeGain, verbose=False)
                greedyIdxs.extend([x[0] for x in greedyListPerClass])
                gammas.extend([1]*budget)

        if self.selection_type in ["PerClass"]:
            rand_indices = np.random.permutation(len(greedyIdxs))
            greedyIdxs = list(np.array(greedyIdxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        smi_end_time = time.time()
        self.logger.info("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return greedyIdxs, gammas
