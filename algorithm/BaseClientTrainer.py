import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import List, Optional, Tuple
import torchvision.transforms as transforms

from .measures import *
# from .dot import DistillationOrientedTrainer

__all__ = ["BaseClientTrainer"]


class DistillationOrientedTrainer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, momentum_kd=0.9, weight_decay=5e-4):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(DistillationOrientedTrainer, self).__init__(params, defaults)
        self.momentum_kd = momentum_kd

        # 记录参数便于后续跟踪
        self.params_list = list(params)

    def step_kd(self, closure=None):
        """处理蒸馏损失梯度"""
        loss = None
        if closure is not None:
            loss = closure()
        # 存储当前梯度
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                    # 获取并存储蒸馏梯度
                d_p = p.grad.data
                # 应用权重衰减
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # 限制梯度大小，防止不稳定
                grad_norm = d_p.norm()
                if grad_norm > 10.0:
                    d_p.mul_(10.0 / grad_norm)

                param_state = self.state[p]
                # 初始化蒸馏动量缓冲区
                if 'kd_momentum_buffer' not in param_state:
                    param_state['kd_momentum_buffer'] = torch.zeros_like(p.data)
                    param_state['kd_update_count'] = 0

                # 计算并存储蒸馏动量梯度
                kd_buf = param_state['kd_momentum_buffer']
                kd_buf.mul_(self.momentum_kd).add_(d_p, alpha=1 - self.momentum_kd)
                # 增加更新计数
                param_state['kd_update_count'] += 1
                # 暂不更新参数，只存储动量
        return loss

    def step(self, closure=None):
        """执行任务更新，结合蒸馏更新"""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                # 获取任务梯度并处理权重衰减
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # 限制梯度大小
                grad_norm = d_p.norm()
                if grad_norm > 10.0:
                    d_p.mul_(10.0 / grad_norm)

                param_state = self.state[p]

                # 初始化任务动量缓冲区
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                # 更新任务动量缓冲区
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1 - momentum)

                # 准备更新：结合任务动量和蒸馏动量（如果可用）
                update = buf.clone()

                if 'kd_momentum_buffer' in param_state and param_state['kd_update_count'] > 0:
                    kd_buf = param_state['kd_momentum_buffer']
                    # 关键步骤：结合任务动量和蒸馏动量
                    update.add_(kd_buf)
                # 应用参数更新
                p.data.add_(update, alpha=-lr)
        return loss

class BaseClientTrainer:
    def __init__(self, algo_params, model, local_epochs, device, num_classes):
        """
        ClientTrainer class for FedSLKD++ algorithm.
        Implements local training with knowledge distillation and group-based updates.
        """
        # 算法特定参数
        self.algo_params = algo_params

        # 模型和训练相关参数
        self.model = model  # 本地模型 (学生模型)
        self.teacher_model = None  # 教师模型 (从服务器接收)
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None
        self.num_classes = num_classes
        self.trainloader = None
        self.testloader = None

        # FedSLKD++特定参数
        self.assigned_group = None  # 分配的组ID
        self.feature_signature = None  # 客户端数据特征签名
        self.computing_capability = 1.0  # 默认计算能力
        self.compression_errors = {} #误差记录

        # DOT蒸馏参数
        self.momentum_task = algo_params.get('momentum_task', 0.9)
        self.delta = algo_params.get('delta', 0.05)
        self.optimizer = None  # 将在receive_global_model中初始化

    def receive_global_model(self, global_model_weights, assigned_group):
        """
        接收服务器发送的全局模型、分配的组ID
        """
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.load_state_dict(global_model_weights)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # 设为评估模式，冻结参数
        # 更新本地学生模型
        self.model.load_state_dict(global_model_weights)
        self.model.to(self.device)
        # current_model_state = self.model.state_dict()

        # 更新分配信息
        self.assigned_group = assigned_group

        # 设置DOT优化器
        # 设置组特定的动量参数
        if assigned_group == 0:  # 浅层组
            m_task = 0.8
            m_kd = 0.9  # 较低的蒸馏动量
        elif assigned_group == 1:  # 中层组
            m_task = 0.85
            m_kd = 0.95
        else:  # 深层组
            m_task = 0.9
            m_kd = 0.95  # 较高的蒸馏动量
        # m_task = self.momentum_task - self.delta
        # m_kd = self.momentum_task + self.delta
        # 创建DOT优化器
        self.optimizer = DistillationOrientedTrainer(
            self.model.parameters(),  # 训练所有参数
            lr=self.algo_params.get('lr', 0.01),
            momentum=m_task,
            momentum_kd=m_kd,
            weight_decay=self.algo_params.get('weight_decay', 5e-4),
        )

    def extract_feature_signature(self, feature_extractor):
        """
        更加健壮的特征提取方法
        """
        # 标准化预处理
        transform = transforms.Compose([
            # 通用的数据转换处理
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.Lambda(lambda x: transforms.ToPILImage()(x) if isinstance(x, torch.Tensor) else x),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        def safe_transform(x):
            """安全的图像转换"""
            try:
                # 处理各种可能的输入类型
                if isinstance(x, tuple):
                    x = x[0]  # 如果是 (image, label) 元组
                # 转换为PIL图像
                if isinstance(x, torch.Tensor):
                    x = transforms.ToPILImage()(x)
                # 应用变换
                return transform(x)
            except Exception as e:
                print(f"图像转换错误: {e}")
                return None
        # 安全地收集有效图像
        valid_images = []
        dataset = self.trainloader.dataset
        for i in range(min(100, len(dataset))):
            img = dataset[i]
            transformed_img = safe_transform(img)
            if transformed_img is not None:
                valid_images.append(transformed_img)
        # 如果没有有效图像，返回None
        if not valid_images:
            print("警告：未能提取任何有效图像特征")
            return None
        # 将图像转换为Tensor批次
        images_tensor = torch.stack(valid_images).to(self.device)
        # 提取特征
        feature_extractor.eval()
        with torch.no_grad():
            batch_features = feature_extractor(images_tensor)
        # 计算特征统计量
        mean_feature = torch.mean(batch_features, dim=0)
        std_feature = torch.std(batch_features, dim=0)
        signature = torch.cat([mean_feature, std_feature]).cpu().numpy()
        return signature

    def train_with_distillation(self):
        self.model.train()
        self.teacher_model.eval()
        # print(f"客户端训练开始 - 分配组: {self.assigned_group}")


        # 保存原始BatchNorm统计量
        # bn_stats = {}
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         bn_stats[name] = {
        #             'running_mean': module.running_mean.clone(),
        #             'running_var': module.running_var.clone(),
        #             'num_batches_tracked': module.num_batches_tracked.clone()
        #         }
        #         # 强制设置评估模式，防止更新统计量
        #         module.eval()

        if self.assigned_group == 0:  # 浅层组
            lr = 0.01
            T = 4.0
            distill_weight = 0.3  # 增加蒸馏权重
            task_weight = 0.7
        elif self.assigned_group == 1:  # 中层组
            lr = 0.01
            T = 5.0
            distill_weight = 0.4
            task_weight = 0.6
        else:  # 深层组
            lr = 0.01
            T = 6.0
            distill_weight = 0.5  # 更侧重蒸馏
            task_weight = 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            samples = 0
            for batch_idx, (batch_x, batch_y) in enumerate(self.trainloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_size = batch_x.size(0)
                samples += batch_size

                # 教师模型前向传播
                with torch.no_grad():
                    teacher_features, teacher_logits = self.teacher_model(batch_x, return_features=True)

                #学生模型前向传播
                student_features, student_logits = self.model(batch_x, return_features=True)
                feature_loss = nn.MSELoss()(student_features, teacher_features)

                distill_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T * T)

                kd_loss =  0.5 * distill_loss + 0.5 * feature_loss
                # kd_loss = distill_weight * kd_loss

                task_loss = nn.CrossEntropyLoss()(student_logits, batch_y)
                # ce_loss = task_weight * task_loss
                # DOT优化步骤1: 处理蒸馏损失
                self.optimizer.zero_grad(set_to_none=True)
                feature_loss.backward(retain_graph=True)
                self.optimizer.step_kd()

                # DOT优化步骤2: 处理任务损失
                self.optimizer.zero_grad(set_to_none=True)
                task_loss.backward()
                self.optimizer.step()

                epoch_loss += feature_loss.item() * batch_size

                # 在第一个batch输出调试信息
                # if batch_idx == 0 and epoch == 0:
                #     print(f"[诊断] 教师logits: {teacher_logits[0, :5].detach().cpu().numpy()}")
                #     print(f"[诊断] 学生logits: {student_logits[0, :5].detach().cpu().numpy()}")
                #     print(f"[诊断] 任务损失: {task_loss.item():.4f}, 蒸馏损失: {kd_loss.item():.4f}")
            #
            # param_norm = sum(p.norm().item() for p in self.model.parameters())
            # print(f"[诊断] Epoch {epoch} 结束, 平均损失: {epoch_loss / samples:.4f}, 参数范数: {param_norm:.4f}")


            # 恢复原始BatchNorm统计量
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.BatchNorm2d) and name in bn_stats:
        #         module.running_mean.copy_(bn_stats[name]['running_mean'])
        #         module.running_var.copy_(bn_stats[name]['running_var'])
        #         module.num_batches_tracked.copy_(bn_stats[name]['num_batches_tracked'])


            # 训练后检查模型变化
        # final_param_norm = sum(p.norm().item() for p in self.model.parameters())
        # print(f"[诊断] 训练后参数范数: {final_param_norm:.4f}")
        #
        # # 计算与教师模型的差异
        # diff_norm = 0.0
        # for p1, p2 in zip(self.model.parameters(), self.teacher_model.parameters()):
        #     diff_norm += (p1 - p2).norm().item()
        # print(f"[诊断] 与教师模型差异: {diff_norm:.4f}")

        # 获取本地训练统计信息
        local_results = self._get_local_stats()

        return local_results, self.datasize

    def compute_and_compress_update(self):
        """只上传分配组和分类器的参数"""
        # 获取分配组的参数更新
        local_group_dict = self.model.get_group_state_dict(self.assigned_group)
        global_group_dict = self.teacher_model.get_group_state_dict(self.assigned_group)

        group_update = {}
        group_norm = 0.0

        for key in local_group_dict.keys():
            if key in global_group_dict:
                update = local_group_dict[key] - global_group_dict[key]
                update = update.float()
                group_update[key] = update
                group_norm += torch.norm(update).item() ** 2

        group_norm = np.sqrt(group_norm)

        # 获取分类器参数更新
        local_classifier_dict = self.model.get_classifier_state_dict()
        global_classifier_dict = self.teacher_model.get_classifier_state_dict()

        classifier_update = {}
        for key in local_classifier_dict.keys():
            if key in global_classifier_dict:
                classifier_update[key] = local_classifier_dict[key] - global_classifier_dict[key]


        compressed_group_update = self._compress_update(group_update, k_ratio=0.1)

        # 统计参数数量
        group_params_count = sum(torch.count_nonzero(v).item() for v in compressed_group_update.values())
        classifier_params_count = sum(v.numel() for v in classifier_update.values())

        update_info={
            'group_update': compressed_group_update,
            'classifier_update': classifier_update,
            'group_norm': group_norm,
            'assigned_group': self.assigned_group,
            # 'skip_update': skip_update,
            'upload_params_count': group_params_count + classifier_params_count,
            'total_model_params': sum(p.numel() for p in self.model.parameters()),
        }
        return update_info




    def _compress_update(self, update_dict, k_ratio=0.1):
        """
        对更新进行Top-k压缩
        Args:
            update_dict: 包含更新的字典
            k_ratio: 保留的比例 (默认0.1，即10%)
        """
        compressed_dict = {}
        for key, tensor in update_dict.items():
            # 如果有上一轮的压缩误差，先加回去
            if key in self.compression_errors:
                tensor = tensor + self.compression_errors[key]
            # 展平张量
            flattened = tensor.view(-1)
            tensor_size = flattened.numel()


            # 计算要保留的元素数量
            k = max(1, int(tensor_size * k_ratio))
            # 计算绝对值
            abs_values = torch.abs(flattened)
            # 找到top-k的索引
            _, indices = torch.topk(abs_values, k)

            # 创建压缩后的张量 (只保留top-k值，其余为0)
            mask = torch.zeros_like(flattened)
            mask[indices] = 1
            # 应用掩码
            compressed = flattened * mask
            #计算误差
            compression_error = flattened - compressed
            self.compression_errors[key] = compression_error.view(tensor.shape)

            # 重塑为原始形状
            compressed_dict[key] = compressed.view(tensor.shape)

        return compressed_dict

    def _get_local_stats(self):
        """获取本地训练统计信息"""
        local_results = {}

        local_results["train_acc"] = evaluate_model(
            self.model, self.trainloader, self.device
        )
        (
            local_results["classwise_accuracy"],
            local_results["test_acc"],
        ) = evaluate_model_classwise(
            self.model, self.testloader, self.num_classes, device=self.device,
        )

        return local_results

    def reset(self):
        """清除现有设置"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.teacher_model = None
        self.assigned_group = None
        # self.group_threshold = None
        self.optimizer = None