from train_sl_siamese import TrainClassifier
import argparse
from cords.utils.config_utils import load_config_data
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="configs/SL/config_full_toy_da5.py")
    parser.add_argument('--smi_func_type', type=str, default='fl2mi')
    parser.add_argument('--query_size', type=int, default=50)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--select_every', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--source_domains', type=str, default="d0")
    parser.add_argument('--target_domains', type=str, default="d1")

    args = parser.parse_args()
    config_file = args.config_file
    config_data = load_config_data(args.config_file)
    # classifier = TrainClassifier(config_file)
    config_data.config_file = config_file
    config_data.dss_args.smi_func_type = args.smi_func_type
    config_data.dss_args.query_size = args.query_size
    config_data.dss_args.fraction = args.fraction
    config_data.dss_args.select_every = args.select_every
    config_data.train_args.device = args.device
    config_data.train_args.print_every = args.print_every
    config_data.train_args.num_epochs = args.num_epochs
    config_data.dss_args.similarity_criterion = args.similarity_criterion
    config_data.dss_args.selection_type = args.selection_type
    if config_data.dataset.name in ["domainnet", "toy_da", "toy_da2", "office31", "officehome", "toy_da3", "toy_da5"]:
        source_domains = args.source_domains.split(",")
        target_domains = args.target_domains.split(",")
        if config_data.dataset.name in ["domainnet", "office31", "officehome"]:
            config_data.dataset.customImageListParams.source_domains = source_domains
            config_data.dataset.customImageListParams.target_domains = target_domains
        else:
            config_data.dataset.daParams.source_domains = source_domains
            config_data.dataset.daParams.target_domains = target_domains
    classifier = TrainClassifier(config_data)
    classifier.train()