from dss_train_sl import TrainClassifier
from dss_train_sl_siamese import TrainClassifier as SiameseClassifier

import os
import argparse
from cords.utils.config_utils import load_config_data
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="configs/SL/config_smi_toy_da.py")
    parser.add_argument('--smi_func_type', type=str, default='fl2mi')
    parser.add_argument('--query_size', type=int, default=100)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--select_every', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--source_domains', type=str, default="real")
    parser.add_argument('--target_domains', type=str, default="clipart")
    parser.add_argument('--similarity_criterion', type=str, default="gradient")
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--selection_type', type=str, default="Supervised")
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--loss', type=str, default=None,
                        help="Loss function: can be CrossEntropyLoss, ccsa, or dsne")
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('--augment_queryset', type=bool, default=False)
    args = parser.parse_args()
    config_file = args.config_file
    config_data = load_config_data(args.config_file)
    # classifier = TrainClassifier(config_file)
    config_data.config_file = config_file
    config_data.dss_args.smi_func_type = args.smi_func_type
    config_data.dss_args.query_size = args.query_size
    config_data.dss_args.fraction = args.fraction
    config_data.dss_args.select_every = args.select_every
    config_data.ckpt.save_every = args.save_every
    if args.loss:
        config_data.loss.type = args.loss
    if args.ckpt_file:
        config_data.ckpt.file = args.ckpt_file
        config_data.ckpt.is_load = True
    config_data.train_args.device = args.device
    config_data.train_args.print_every = args.print_every
    config_data.train_args.num_epochs = args.num_epochs
    config_data.dss_args.similarity_criterion = args.similarity_criterion
    config_data.dss_args.selection_type = args.selection_type
    if config_data.dataset.name in ["domainnet", "toy_da", "toy_da2", "office31", "officehome", "toy_da3"]:
        source_domains = args.source_domains.split(",")
        target_domains = args.target_domains.split(",")
        if config_data.dataset.name in ["domainnet", "office31", "officehome"]:
            config_data.dataset.customImageListParams.source_domains = source_domains
            config_data.dataset.customImageListParams.target_domains = target_domains
        else:
            config_data.dataset.daParams.source_domains = source_domains
            config_data.dataset.daParams.target_domains = target_domains
    if config_data.loss.type == 'ccsa':
        config_data.model.architecture = 'SiameseResNet50'
        config_data.train_args.alpha = 0.25
    if config_data.loss.type == 'dsne':
        config_data.model.architecture = 'SiameseResNet50'
        config_data.train_args.alpha = 0.1
    for i in range(args.num_runs):
        print(f"RUN: {i}")
        config = config_data.copy()
        config['run_id'] = i
        if config_data.loss.type in ['ccsa', 'dsne']:
            config_data.dss_args.augment_queryset = False
            config_data.train_args.train_type = 'ft' if args.fine_tune else ''
            config_data.train_args.ft_epochs = 5 if args.fine_tune else 0
            classifier = SiameseClassifier(config)
        else:
            config_data.dss_args.fine_tune = args.fine_tune
            config_data.dss_args.augment_queryset = args.augment_queryset
            classifier = TrainClassifier(config)
        classifier.train()