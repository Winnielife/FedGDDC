import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small,resnet18,efficientnet_b0

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# 导入特征提取器模型定义
# from .models.model import FeatureExtractor


def load_pretrained_feature_extractor(
    model_name='resnet18',
    pretrained=True,
    device="cuda"
):
    print(f"加载PyTorch官方预训练的 {model_name} 模型...")

    # 根据模型名称选择不同的预训练模型
    if model_name == 'mobilenet_v3_small':
        pretrained_model = mobilenet_v3_small(pretrained=pretrained)
        feature_extractor = nn.Sequential(
            pretrained_model.features,
            pretrained_model.avgpool,
            nn.Flatten()
        )
    elif model_name == 'resnet18':
        pretrained_model = resnet18(pretrained=pretrained)
        feature_extractor = nn.Sequential(
            nn.Sequential(
                pretrained_model.conv1,
                pretrained_model.bn1,
                pretrained_model.relu,
                pretrained_model.maxpool
            ),
            pretrained_model.layer1,
            pretrained_model.layer2,
            pretrained_model.layer3,
            pretrained_model.layer4,
            pretrained_model.avgpool,
            nn.Flatten()
        )
    elif model_name == 'efficientnet_b0':
        pretrained_model = efficientnet_b0(pretrained=pretrained)
        feature_extractor = nn.Sequential(
            pretrained_model.features,
            pretrained_model.avgpool,
            nn.Flatten()
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 冻结特征提取器的参数
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # 将特征提取器移到指定设备
    feature_extractor = feature_extractor.to(device)

    # 设置为评估模式
    feature_extractor.eval()

    print("特征提取器加载完成。")

    return feature_extractor


# class NTXentLoss(nn.Module):
#     """对比损失函数实现"""
#
#     def __init__(self, temperature=0.5):
#         super().__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, z_i, z_j):
#         batch_size = z_i.size(0)
#
#         # 计算相似度矩阵
#         z = torch.cat([z_i, z_j], dim=0)
#         z = F.normalize(z, dim=1)
#         similarity_matrix = torch.matmul(z, z.T) / self.temperature
#
#         # 排除自身相似度
#         sim_i_j = torch.diag(similarity_matrix, batch_size)
#         sim_j_i = torch.diag(similarity_matrix, -batch_size)
#
#         # 正样本对
#         positives = torch.cat([sim_i_j, sim_j_i], dim=0)
#
#         # 构建标签
#         labels = torch.arange(batch_size, device=z_i.device)
#         labels = torch.cat([labels, labels], dim=0)
#
#         # 移除对角线元素，消除自身相似度
#         mask = torch.ones_like(similarity_matrix) - torch.eye(2 * batch_size, device=z_i.device)
#
#         # 掩码相似度矩阵，移除自身相似度
#         similarity_matrix = similarity_matrix * mask
#
#         # 选择正样本对
#         positives = torch.exp(positives)
#         negatives = torch.sum(torch.exp(similarity_matrix), dim=1)
#
#         # 计算损失
#         loss = -torch.log(positives / negatives)
#         return loss.mean()
#
#
# def get_simclr_data_transforms():
#     """定义SimCLR数据增强"""
#     color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
#
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([color_jitter], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     return train_transform
#
#
# class TwoCropsTransform:
#     """Take two random crops of one image"""
#
#     def __init__(self, base_transform):
#         self.base_transform = base_transform
#
#     def __call__(self, x):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#         return [q, k]
#
#
# def pretrain_feature_extractor(model_save_path='./models/pretrained_feature_extractor.pth', data_path='./data/imagenet'):
#     """预训练特征提取器"""
#     print("开始预训练特征提取器...")
#
#     # 检查预训练模型是否已存在
#     import os
#     if os.path.exists(model_save_path):
#         print(f"预训练模型已存在于 {model_save_path}，跳过预训练")
#         model = FeatureExtractor(output_dim=128)
#         model.load_state_dict(torch.load(model_save_path))
#         # 移除投影头，仅保留编码器
#         model.projection = nn.Identity()
#         return model
#
#     # 数据加载和增强
#     train_transform = get_simclr_data_transforms()
#     train_dataset = datasets.ImageFolder(
#         os.path.join(data_path, 'train'),
#         TwoCropsTransform(train_transform)
#     )
#
#     train_loader = DataLoader(
#         train_dataset, batch_size=256, shuffle=True,
#         num_workers=8, pin_memory=True, drop_last=True
#     )
#
#     # 初始化模型
#     model = FeatureExtractor(output_dim=128)
#
#     # 移到GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#
#     # SimCLR损失函数
#     criterion = NTXentLoss(temperature=0.5)
#
#     # 优化器
#     optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#
#     # 训练循环
#     num_epochs = 150
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0.0
#
#         for i, ((x_i, x_j), _) in enumerate(train_loader):
#             x_i, x_j = x_i.to(device), x_j.to(device)
#
#             # 获取两个视图的表征
#             _, z_i = model(x_i, return_projection=True)
#             _, z_j = model(x_j, return_projection=True)
#
#             # 计算对比损失
#             loss = criterion(z_i, z_j)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#             if (i + 1) % 20 == 0:
#                 print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
#         avg_loss = total_loss / len(train_loader)
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
#
#     # 保存模型
#     torch.save(model.state_dict(), model_save_path)
#     print(f"预训练完成，模型保存至 {model_save_path}")
#
#     # 移除投影头，仅保留编码器
#     model.projection = nn.Identity()
#
#     return model
#
#
# if __name__ == "__main__":
#     # 直接运行此脚本可以预训练模型
#     pretrain_feature_extractor()