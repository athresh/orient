import math
import numpy as np
import time
import torch
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
from submodlib.submodlib.functions import facilityLocationMutualInformation

class dss_ds_strategy(DataSelectionStrategy):
    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex,
                 selection_type, logger, optimizer='lazy'):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.logger = logger
        self.optimizer = optimizer

    def compute_score(self, model_params, idxs):
        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                    sampler=SubsetRandomSampler(idxs), pin_memory=True)
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        if self.if_convex:
            for batch_idx, batch in enumerate(subset_loader):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    inputs, targets, domain = batch
                if self.selection_type == 'Supervised':
                    self.N += inputs.size()[0]
                    g_is.append(inputs.view(inputs.size()[0], -1))

        self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
        first_i = True
        if self.selection_type == 'Supervised':
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def get_similarity_kernel(self):
        """
        Obtain the similarity kernel.

        Returns
        ----------
        kernel: ndarray
            Array of kernel values
        """
        for batch_idx, batch in enumerate(self.trainloader):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, domain = batch
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        kernel = np.zeros((labels.shape[0], labels.shape[0]))
        for target in np.unique(labels):
            x = np.where(labels == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def select(self, budget, model_params):
        """

        Parameters
        ----------
        budget :
        model_params :

        Returns
        -------

        """
        start_time = time.time()
        for batch_idx, batch in enumerate(self.trainloader):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, domain = batch

            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'Supervised':
            for i in range(self.num_classes):
                if i == 0:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = idxs.repeat_interleave(N)
                    col = idxs.repeat(N)
                    data = self.dist_mat.flatten()
                else:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                    col = torch.cat((col, idxs.repeat(N)), dim=0)
                    data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
            sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
            self.dist_mat = sparse_simmat
            fl = facilityLocationMutualInformation.FacilityLocationMutualInformationFunction()