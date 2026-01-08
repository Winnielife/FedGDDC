import torch.nn as nn
import torch

import torchvision.models as models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super(ResNet, self).__init__()

        # 初始层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 特征提取器 - 按深度分为3组
        self.group0 = nn.Sequential(  # 浅层组
            BasicBlock(32, 32, 1),
            BasicBlock(32, 64, 2)
        )

        self.group1 = nn.Sequential(  # 中层组
            BasicBlock(64, 64, 1),
            BasicBlock(64, 128, 2)
        )

        self.group2 = nn.Sequential(  # 深层组
            BasicBlock(128, 128, 1),
            BasicBlock(128, 256, 2)
        )

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.feature_transform = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        """完整前向传播 - 永远执行所有层"""
        # 初始层
        x = torch.relu(self.bn1(self.conv1(x)))

        # 所有组依次执行
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)

        # 分类器
        x = self.avgpool(x)
        x = self.flatten(x)
        features = self.feature_transform(x)
        logits = self.classifier(features)

        if return_features:
            return features, logits
        return logits

    def get_group_state_dict(self, group_id):
        """获取指定组的状态字典"""
        group_dict = {}

        if group_id == 0:
            prefix = 'group0'
        elif group_id == 1:
            prefix = 'group1'
        elif group_id == 2:
            prefix = 'group2'
        else:
            raise ValueError(f"Invalid group_id: {group_id}")

        full_state = self.state_dict()
        for key, value in full_state.items():
            if key.startswith(prefix):
                group_dict[key] = value

        return group_dict

    def get_classifier_state_dict(self):
        """获取分类器状态字典"""
        classifier_dict = {}
        full_state = self.state_dict()

        for key, value in full_state.items():
            if key.startswith('feature_transform') or key.startswith('classifier'):
                classifier_dict[key] = value

        return classifier_dict
