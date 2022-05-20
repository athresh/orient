import numpy as np
from sklearn.manifold import TSNE
import torch
from cords.utils.models import *


def create_model(cfg):
    if cfg.model.architecture == 'ResNet18':
        model = ResNet18(cfg.model.numclasses)
    elif cfg.model.architecture == 'MnistNet':
        model = MnistNet()
    elif cfg.model.architecture == 'ResNet164':
        model = ResNet164(cfg.model.numclasses)
    elif cfg.model.architecture == 'ResNet50':
        if cfg.model.pretrained:
            model = ResNetPretrained('ResNet50', class_num=cfg.model.numclasses)
        else:
            model = ResNet50(cfg.model.numclasses)
    elif cfg.model.architecture == 'MobileNet':
        model = MobileNet(cfg.model.numclasses)
    elif cfg.model.architecture == 'MobileNetV2':
        model = MobileNetV2(cfg.model.numclasses)
    elif cfg.model.architecture == 'MobileNet2':
        model = MobileNet2(output_size=self.cfg.model.numclasses)
    elif cfg.model.architecture == 'HyperParamNet':
        model = HyperParamNet(cfg.model.l1, self.cfg.model.l2)
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


@staticmethod
def load_ckpt(ckpt_path, model, optimizer):
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    metrics = checkpoint['metrics']
    return start_epoch, model, optimizer, loss, metrics

if __name__ == '__main__':
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