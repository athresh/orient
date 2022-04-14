'''ResNet in PyTorch.

Reference
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SiameseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SiameseResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 512
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.embed1 = nn.Linear(512 * block.expansion, 4096)
        self.embed2 = nn.Linear(4096, 512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False, feature=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
                e = self.embed1(e)
                e = self.embed2(e)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            e = out.view(out.size(0), -1)
            e = self.embed1(e)
            e = self.embed2(e)
        out = self.linear(e)
        if last:
            if feature:
                return out, e, e
            else:
                return out, e
        else:
            if feature:
                return out, e
            else:
                return out

    def get_embedding_dim(self):
        return self.embDim


def SiameseResNet18(num_classes=10):
    return SiameseResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def SiameseResNet34(num_classes=10):
    return SiameseResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def SiameseResNet50(num_classes=10):
    return SiameseResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def SiameseResNet101(num_classes=10):
    return SiameseResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def SiameseResNet152(num_classes=10):
    return SiameseResNet(Bottleneck, [3, 8, 36, 3], num_classes)


resnet_dict = {'ResNet18': models.resnet18, 'ResNet34': models.resnet34, 'ResNet50': models.resnet50,
               'ResNet101': models.resnet101, 'ResNet152': models.resnet152}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class SiameseResNetPretrained(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=False, bottleneck_dim=256, new_cls=True, class_num=1000):
        super(SiameseResNetPretrained, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
                                            self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        # if new_cls:
        if self.use_bottleneck:
            self.embed1 = nn.Linear(model_resnet.fc.in_features, 4096)
            self.embed2 = nn.Linear(4096, 512)
            self.bottleneck = nn.Linear(512, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.embed1.apply(init_weights)
            self.embed2.apply(init_weights)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.embed1 = nn.Linear(model_resnet.fc.in_features, 4096)
            self.embed2 = nn.Linear(4096, 512)
            self.embed1.apply(init_weights)
            self.embed2.apply(init_weights)
            self.fc = nn.Linear(512, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 512
        # else:
        #     self.fc = model_resnet.fc
        #     self.__in_features = model_resnet.fc.in_features

    def forward_one(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.embed1(x)
        x = self.embed2(x)
        return x

    def forward(self, x, feature=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                embed = self.forward_one(x)
                if self.use_bottleneck and self.new_cls:
                    embed1 = self.bottleneck(embed)
        else:
            embed = self.forward_one(x)
            if self.use_bottleneck:
                embed1 = self.bottleneck(embed)
        if self.use_bottleneck:
            y = self.fc(embed1)
        else:
            y = self.fc(embed)

        if last:
            if feature:
                return y, x, embed
            else:
                return y, x
        else:
            if feature:
                return y, embed
            else:
                return y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
            else:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
        else:
            parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list

    def get_embedding_dim(self):
        return self.fc.in_features

# for test
if __name__ == '__main__':
    net = SiameseResNetPretrained('ResNet18', use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1000)
    print(net)
    print(list(net.parameters()))