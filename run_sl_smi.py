from train_sl import TrainClassifier
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
    if config_data.dataset.name in ["domainnet", "toy_da", "office31", "officehome"]:
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