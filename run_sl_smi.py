from train_sl import TrainClassifier
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="configs/SL/config_smi_civilcomments.py")
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--select_every', type=int, default=2)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    config_file = args.config_file
    classifier = TrainClassifier(config_file)
    classifier.cfg.dss_args.fraction = args.fraction
    classifier.cfg.dss_args.select_every = args.select_every
    classifier.cfg.train_args.device = args.device
    classifier.cfg.train_args.print_every = args.print_every
    classifier.train()