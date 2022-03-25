import logging
import os
import os.path as osp
import sys
import time
import torch
from reggol import setup_logger
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, OLRandomDataLoader, \
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader, SMIDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
import matplotlib.pyplot as plt


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)


def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.
    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.
    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


class TrainClassifier:
    def __init__(self, config_data):
        version = config_data.dss_args.type
        if config_data.dss_args.type == 'SMI':
            version += f"_{config_data.dss_args.smi_func_type}"
        exp_domains = f"{'-'.join(config_data.dataset.customImageListParams.source_domains)}_vs_{'-'.join(config_data.dataset.customImageListParams.target_domains)}"
        config_data['exp_domains'] = exp_domains
        config_data['version'] = version
        self.logger = setup_logger(f"{version}_{exp_domains}", config_data, exp_id=os.getpid(),
                                   snapshot_gap=config_data.ckpt.save_every)
        self.cfg = config_data
        if "toy_da" in self.cfg.dataset.name:
            self.da_dir_extension = str(self.cfg.dataset.daParams.source_domains) + '->' + str(
                self.cfg.dataset.daParams.target_domains)
        else:
            self.da_dir_extension = str(self.cfg.dataset.customImageListParams.source_domains) + '->' + str(
                self.cfg.dataset.customImageListParams.target_domains)

    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    inputs, targets, domains = batch
                else:
                    raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                                  targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    def eval_group(self, dataset, y_pred, y_true, metadata, prediction_fn=None):
        from wilds.common.metrics.all_metrics import Accuracy
        metric = Accuracy(prediction_fn=prediction_fn)
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        # Each eval_grouper is over label + a single identity
        # We only want to keep the groups where the identity is positive
        # The groups are:
        #   Group 0: identity = 0, y = 0
        #   Group 1: identity = 1, y = 0
        #   Group 2: identity = 0, y = 1
        #   Group 3: identity = 1, y = 1
        # so this means we want only groups 1 and 3.
        worst_group_metric = None
        for identity_var, eval_grouper in zip(dataset._identity_vars, dataset._eval_groupers):
            g = move_to(eval_grouper.metadata_to_group(metadata), self.cfg.train_args.device)
            group_results = {
                **metric.compute_group_wise(y_pred, y_true, g, eval_grouper.n_groups)
            }
            results_str += f"  {identity_var:20s}"
            for group_idx in range(eval_grouper.n_groups):
                group_str = eval_grouper.group_field_str(group_idx)
                if f'{identity_var}:1' in group_str:
                    group_metric = group_results[metric.group_metric_field(group_idx)]
                    group_counts = group_results[metric.group_count_field(group_idx)]
                    results[f'{metric.name}_{group_str}'] = group_metric
                    results[f'count_{group_str}'] = group_counts
                    if f'y:0' in group_str:
                        label_str = 'non_toxic'
                    else:
                        label_str = 'toxic'
                    results_str += (
                        f"   {metric.name} on {label_str}: {group_metric:.3f}"
                        f" (n = {results[f'count_{group_str}']:6.0f}) "
                    )
                    if worst_group_metric is None:
                        worst_group_metric = group_metric
                    else:
                        worst_group_metric = metric.worst(
                            [worst_group_metric, group_metric])
            results_str += f"\n"
        results[f'{metric.worst_group_metric_field}'] = worst_group_metric
        results_str += f"Worst-group {metric.name}: {worst_group_metric:.3f}\n"

        return results, results_str

    """
    ############################## Model Creation ##############################
    """

    def create_model(self):
        if self.cfg.model.architecture == 'ResNet18':
            model = ResNet18(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MnistNet':
            model = MnistNet()
        elif self.cfg.model.architecture == 'ResNet164':
            model = ResNet164(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'ResNet50':
            if self.cfg.model.pretrained:
                model = ResNetPretrained('ResNet50', class_num=self.cfg.model.numclasses)
            else:
                model = ResNet50(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet':
            model = MobileNet(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNetV2':
            model = MobileNetV2(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet2':
            model = MobileNet2(output_size=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'HyperParamNet':
            model = HyperParamNet(self.cfg.model.l1, self.cfg.model.l2)
        elif self.cfg.model.architecture == 'logreg_net':
            model = LogisticRegNet(self.cfg.model.numclasses, self.cfg.model.input_dim)
        elif self.cfg.model.architecture == 'distilbert':
            model = DistilBertClassifier.from_pretrained('distilbert-base-uncased',
                                                         num_labels=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'TwoLayerNet':
            model = TwoLayerNet(self.cfg.model.input_dim, self.cfg.model.numclasses,
                                hidden_units=self.cfg.model.hidden_units)
        elif self.cfg.model.architecture == 'ThreeLayerNet':
            model = ThreeLayerNet(self.cfg.model.input_dim, self.cfg.model.numclasses, h1=self.cfg.model.h1,
                                  h2=self.cfg.model.h2)
        model = model.to(self.cfg.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """

    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.cfg.optimizer.type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.optimizer.lr,
                                  momentum=self.cfg.optimizer.momentum,
                                  weight_decay=self.cfg.optimizer.weight_decay,
                                  nesterov=self.cfg.optimizer.nesterov)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=self.cfg.optimizer.lr,
                                    weight_decay=self.cfg.optimizer.weight_decay)

        if self.cfg.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg.scheduler.T_max)
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing / 3600

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def train(self):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        if self.cfg.dataset.feature == 'classimb':
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature,
                                                               classimb_ratio=self.cfg.dataset.classimb_ratio)
        elif self.cfg.dataset.name in ["office31", "domainnet", "officehome"]:
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature,
                                                               imagelist_params=self.cfg.dataset.customImageListParams,
                                                               preprocess_params=self.cfg.dataset.preprocess,
                                                               augment_queryset=self.cfg.dss_args.augment_queryset)
        elif "toy_da" in self.cfg.dataset.name:
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature,
                                                               daParams=self.cfg.dataset.daParams)
        else:
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name,
                                                               self.cfg.dataset.feature)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = 1000

        # Creating the Data Loaders
        if self.cfg.dataset.name in ['civilcomments']:
            from wilds.common.data_loaders import get_train_loader, get_eval_loader
            from wilds.common.metrics.all_metrics import Accuracy
            trainloader = get_train_loader(loader='standard', dataset=trainset, batch_size=trn_batch_size,
                                           pin_memory=True)
            valloader = get_eval_loader(loader='standard', dataset=validset, batch_size=val_batch_size,
                                        pin_memory=True)
            testloader = get_eval_loader(loader='standard', dataset=testset, batch_size=tst_batch_size,
                                         pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                      shuffle=False, pin_memory=True)

            valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                    shuffle=False, pin_memory=True)

            testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                     shuffle=False, pin_memory=True)

        substrn_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = list()
        trn_acc = list()
        val_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        subtrn_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        group_metric = list()

        checkpoint_path = self.cfg.ckpt.file
        # Model Creation
        model = self.create_model()
        # model1 = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if self.cfg.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = GradMatchDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                             batch_size=self.cfg.dataloader.batch_size,
                                             shuffle=self.cfg.dataloader.shuffle,
                                             pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type in ['GLISTER', 'GLISTER-Warm', 'GLISTERPB', 'GLISTERPB-Warm']:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            dataloader = GLISTERDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = CRAIGDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                         batch_size=self.cfg.dataloader.batch_size,
                                         shuffle=self.cfg.dataloader.shuffle,
                                         pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = RandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type == ['OLRandom', 'OLRandom-Warm']:
            """
            ############################## OLRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = OLRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type == 'Full':
            """
            ############################## Full Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))

            dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     shuffle=self.cfg.dataloader.shuffle,
                                                     pin_memory=self.cfg.dataloader.pin_memory)

        elif self.cfg.dss_args.type == 'SMI':
            """

            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = SMIDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                       batch_size=self.cfg.dataloader.batch_size,
                                       shuffle=self.cfg.dataloader.shuffle,
                                       pin_memory=self.cfg.dataloader.pin_memory)
        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckpt(checkpoint_path, model, optimizer)
            logger.info("Loading saved checkpoint model at epoch: {0:d}".format(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """

        for epoch in range(start_epoch, self.cfg.train_args.num_epochs):
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            if self.cfg.train_args.visualize and (epoch + 1) % self.cfg.dss_args.select_every == 0:
                plt.figure()
            # for _, (inputs, targets, domains, weights) in enumerate(dataloader):
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, targets, weights = batch
                elif len(batch) == 4:
                    inputs, targets, domains, weights = batch
                else:
                    raise ValueError("Batch length must be either 3 or 4, not {}".format(len(batch)))
                #             for _, (inputs, targets, weights) in enumerate(dataloader):
                inputs = inputs.to(self.cfg.train_args.device)
                targets = targets.to(self.cfg.train_args.device, non_blocking=True)
                weights = weights.to(self.cfg.train_args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets)
                loss = torch.dot(losses, weights / (weights.sum()))
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
                if self.cfg.train_args.visualize and (epoch + 1) % self.cfg.dss_args.select_every == 0:
                    plt.scatter(inputs.cpu().numpy()[:, 0], inputs.cpu().numpy()[:, 1], marker='o',
                                c=targets.cpu().numpy(),
                                s=25, edgecolor='k')
                # if self.cfg.dataset.name in ["toy_da"]:
                #     for idx in range(len(inputs.cpu().numpy()[:,0])):
                #         if inputs.cpu().numpy()[idx, 0] ==
            if self.cfg.train_args.visualize and (epoch + 1) % self.cfg.dss_args.select_every == 0:
                plt.title(
                    "Strategy: {}({}), Fraction: {}".format(self.cfg.dss_args.type, self.cfg.dss_args.smi_func_type,
                                                            self.cfg.dss_args.fraction))
                if self.cfg.dataset.name == 'toy_da3':
                    plt.xlim(-2.0, 5.0)
                    plt.ylim(-1.0, 2.0)
                else:
                    plt.xlim(-2.0, 2.5)
                    plt.ylim(-1.0, 2.0)
                plt.savefig(self.all_plots_dir + "/selected_data_{}.png".format(epoch))
                # HK: For unsupervised add psuedo labels to valdataloader.
            epoch_time = time.time() - start_time
            scheduler.step()
            timing.append(epoch_time)
            print_args = self.cfg.train_args.print_args

            """
            ################################################# Evaluation Loop #################################################
            """

            if (epoch + 1) % self.cfg.train_args.print_every == 0:
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()

                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    with torch.no_grad():
                        # for _, (inputs, targets, domains) in enumerate(trainloader):
                        for batch_idx, batch in enumerate(trainloader):
                            if len(batch) == 2:
                                inputs, targets = batch
                            elif len(batch) == 3:
                                inputs, targets, domains = batch
                            else:
                                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                            #                         for _, (inputs, targets) in enumerate(trainloader):
                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += loss.item()
                            if "trn_acc" in print_args:
                                _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                        trn_losses.append(trn_loss)
                        logger.record_tabular("Training Loss", trn_losses[-1])

                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)
                        logger.record_tabular("Training Accuracy", trn_acc[-1])

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    with torch.no_grad():
                        # for _, (inputs, targets, domains) in enumerate(valloader):
                        for batch_idx, batch in enumerate(valloader):
                            if len(batch) == 2:
                                inputs, targets = batch
                            elif len(batch) == 3:
                                inputs, targets, domains = batch
                            else:
                                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                            #                         for _, (inputs, targets) in enumerate(valloader):
                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            if "val_acc" in print_args:
                                _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        val_losses.append(val_loss)
                        logger.record_tabular("Validation Loss", val_losses[-1])

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)
                        logger.record_tabular("Validation Accuracy", val_acc[-1])

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    with torch.no_grad():
                        # for _, (inputs, targets, domains) in enumerate(testloader):
                        for batch_idx, batch in enumerate(testloader):
                            if len(batch) == 2:
                                inputs, targets = batch
                            elif len(batch) == 3:
                                inputs, targets, domains = batch
                            else:
                                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                            #                         for _, (inputs, targets) in enumerate(testloader):
                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += loss.item()
                            if "tst_acc" in print_args:
                                _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()

                        tst_losses.append(tst_loss)
                        logger.record_tabular("Test Loss", tst_losses[-1])
                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)
                        logger.record_tabular("Test Accuracy", tst_acc[-1])

                if ("worst_acc" in print_args):
                    with torch.no_grad():
                        val_pred = []
                        val_true = []
                        val_metadata = []
                        tst_pred = []
                        tst_true = []
                        tst_metadata = []
                        # for _, (inputs, targets, domains) in enumerate(valloader):
                        for batch_idx, batch in enumerate(valloader):
                            if len(batch) == 2:
                                inputs, targets = batch
                            elif len(batch) == 3:
                                inputs, targets, domains = batch
                            else:
                                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)
                            val_pred.append(detach_and_clone(predicted))
                            val_true.append(detach_and_clone(targets))
                            val_metadata.append(detach_and_clone(domains))

                        # for _, (inputs, targets, domains) in enumerate(testloader):
                        for batch_idx, batch in enumerate(testloader):
                            if len(batch) == 2:
                                inputs, targets = batch
                            elif len(batch) == 3:
                                inputs, targets, domains = batch
                            else:
                                raise ValueError("Batch length must be either 2 or 3, not {}".format(len(batch)))
                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)
                            tst_pred.append(detach_and_clone(predicted))
                            tst_true.append(detach_and_clone(targets))
                            tst_metadata.append(detach_and_clone(domains))
                        val_pred = collate_list(move_to(val_pred, torch.device('cpu')))
                        val_true = collate_list(move_to(val_true, torch.device('cpu')))
                        val_metadata = collate_list(move_to(val_metadata, torch.device('cpu')))
                        tst_pred = collate_list(move_to(tst_pred, torch.device('cpu')))
                        tst_true = collate_list(move_to(tst_true, torch.device('cpu')))
                        tst_metadata = collate_list(move_to(tst_metadata, torch.device('cpu')))
                        if val_pred.is_cuda:
                            logger.info("val_pred on device: " + str(val_pred.get_device()))
                        if val_true.is_cuda:
                            logger.info("val_true on device: " + str(val_true.get_device()))
                        if val_metadata.is_cuda:
                            logger.info("val_metadata on device: " + str(val_metadata.get_device()))
                        results_val, results_str_val = validset.eval(val_pred, val_true, val_metadata)
                        results_tst, results_str_tst = testset.eval(tst_pred, tst_true, tst_metadata)

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)
                    logger.record_tabular("Subset Accuracy", subtrn_acc[-1])

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)
                    logger.record_tabular("Subset Loss", subtrn_losses[-1])

                print_str = "Epoch: " + str(epoch + 1)
                logger.record_tabular("epoch", epoch + 1)
                logger.record_tabular("Timing", timing[-1])
                logger.record_tabular("Total timing", np.sum(timing))

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.cfg and self.cfg.report_tune:
                    tune.report(mean_accuracy=val_acc[-1])

                logger.dump_tabular()
            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

                metric_dict = {}

                if "val_loss" in print_args:
                    metric_dict['val_loss'] = val_losses
                if "val_acc" in print_args:
                    metric_dict['val_acc'] = val_acc
                if "tst_loss" in print_args:
                    metric_dict['tst_loss'] = tst_losses
                if "tst_acc" in print_args:
                    metric_dict['tst_acc'] = tst_acc
                if "trn_loss" in print_args:
                    metric_dict['trn_loss'] = trn_losses
                if "trn_acc" in print_args:
                    metric_dict['trn_acc'] = trn_acc
                if "subtrn_loss" in print_args:
                    metric_dict['subtrn_loss'] = subtrn_losses
                if "subtrn_acc" in print_args:
                    metric_dict['subtrn_acc'] = subtrn_acc
                if "time" in print_args:
                    metric_dict['time'] = timing

                ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, logger.get_snapshot_dir() + f"/model_{epoch + 1}.pt")
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))
