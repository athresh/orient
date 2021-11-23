from train_sl import TrainClassifier
config_file = "configs/SL/config_random_toy.py"
classifier = TrainClassifier(config_file)
classifier.cfg.dss_args.fraction = 0.1
# classifier.cfg.dss_args.select_every = 2
classifier.cfg.train_args.device = 'cuda'
classifier.cfg.train_args.print_every = 1
classifier.train()