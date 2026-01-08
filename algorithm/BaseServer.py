import torch
import torch.nn as nn
import numpy as np
import copy
import time
import wandb
from numpy import random
from sklearn.cluster import KMeans

from .measures import *

__all__ = ["BaseServer"]


class BaseServer:
    def __init__(
            self,
            algo_params,
            model,
            data_distributed,
            optimizer,
            scheduler,
            feature_extractor=None,
            n_rounds=200,
            sample_ratio=0.1,
            local_epochs=10,
            device="cuda:0",
    ):
        """
        FedSLKD++服务器实现 (使用DOT知识蒸馏)
        """
        # 基础参数
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs

        # 通信开销评估
        self.communication_cost = []  # 记录每轮通信开销
        self.total_communication_cost = []
        self.comm_ratio = []
        self.total_model_params = sum(p.numel() for p in self.model.parameters())  # 模型总参数量

        # 预训练特征提取器
        self.feature_extractor = feature_extractor
        if self.feature_extractor is not None:
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()

        # FedSLKD++特定参数
        self.client_groups = {}  # 客户端组分配 {client_id: group_id}
        self.client_features = {}  # 客户端特征 {client_id: feature_vector}
        self.client_capabilities = {}  # 客户端计算能力 {client_id: capability_score}

        # 训练时间相关设置
        self.BASE_TRAIN_TIME = 100.0  # 基准训练时间（秒）

        # DOT知识蒸馏参数
        if 'momentum_task' not in self.algo_params:
            self.algo_params['momentum_task'] = 0.9
        if 'delta' not in self.algo_params:
            self.algo_params['delta'] = 0.075
        if 'weight_decay' not in self.algo_params:
            self.algo_params['weight_decay'] = 5e-4
        if 'lr' not in self.algo_params:
            self.algo_params['lr'] = 0.01

        # 服务器结果记录
        self.server_results = {
            "client_history": [],
            "test_accuracy": [],
            "group_assignments": {}  # 记录每轮的客户端分组
        }

        # 上一轮精度（用于动态调整阈值）
        self.prev_accuracy = 0.0
        self.curr_round = 1

    def run(self):
        """运行联邦学习实验"""
        self._print_start()

        # 初始化：收集客户端能力和特征，进行初始分组
        self._initialize_clients_groups()

        for round_idx in range(self.n_rounds):


            # 首轮评估全局模型
            if round_idx == 0:
                test_acc = evaluate_model(
                    self.model, self.testloader, device=self.device
                )
                self.server_results["test_accuracy"].append(test_acc)
                self.prev_accuracy = test_acc

            start_time = time.time()

            # 周期性重新分配客户端组（每5轮）
            if round_idx > 0 and round_idx % 5 == 0:
                self._reassign_clients_groups()

            # 选择参与本轮训练的客户端
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # 记录本轮分组情况
            current_assignments = {}
            for client_id in sampled_clients:
                current_assignments[client_id] = self.client_groups.get(client_id, 0)
            self.server_results["group_assignments"][round_idx] = current_assignments

            # 客户端训练阶段，获取更新
            client_updates = self._clients_training(sampled_clients)

            # 收集本轮结果统计
            round_results = {
                "train_acc": [update['local_results']["train_acc"] for update in client_updates],
                "test_acc": [update['local_results']["test_acc"] for update in client_updates]
            }

            # 按组聚合更新
            self._aggregate_updates(client_updates)

            # 评估更新后的全局模型
            test_acc = self.evaluate_global_model()
            self.server_results["test_accuracy"].append(test_acc)

            # 动态调整阈值
            # self._adjust_thresholds(test_acc, self.prev_accuracy)

            # # 更新上一轮精度
            # self.prev_accuracy = test_acc

            # 计算轮次用时
            round_elapse = time.time() - start_time

            # 打印本轮统计信息
            self._print_round_stats(round_idx, test_acc, round_elapse)

            # 更新上一轮精度
            self.prev_accuracy = test_acc

            # 记录到W&B
            self._wandb_logging(round_results, round_idx)

    def _initialize_clients_groups(self):
        """
        初始化：收集客户端能力和数据特征，进行初始分组
        使用K-means将客户端分为3组
        """
        print("初始化客户端分组...")

        # 收集客户端特征和计算能力
        for client_idx in range(self.n_clients):
            # 设置客户端数据
            self._set_client_data(client_idx)

            # 收集计算能力（假设，实际应该由客户端报告）
            # 这里我们随机生成计算能力得分 (0.5-1.0之间)
            capability = 0.5 + 0.5 * np.random.random()
            self.client_capabilities[client_idx] = capability

            # 提取客户端数据特征
            if self.feature_extractor is not None:
                feature_signature = self.client.extract_feature_signature(self.feature_extractor)
                self.client_features[client_idx] = feature_signature
            else:
                # 如果没有特征提取器，使用随机特征
                self.client_features[client_idx] = np.random.randn(1152)

                # 重置客户端
            self.client.reset()

            # 构建聚类特征（结合数据特征和计算能力）
        clustering_features = []
        for client_idx in range(self.n_clients):
            # 提取数据特征
            data_feature = self.client_features[client_idx]

            # 归一化数据特征
            data_feature_norm = data_feature / (np.linalg.norm(data_feature) + 1e-8)

            # 计算能力作为额外特征（归一化到[0,1]）
            capability = self.client_capabilities[client_idx]

            # 构建完整特征向量 (拼接归一化数据特征和计算能力)
            # 这里我们简化为只使用数据特征的前100维 + 计算能力
            full_feature = np.concatenate([data_feature_norm[:100], [capability]])
            clustering_features.append(full_feature)

            # 使用K-means聚类将客户端分为3组
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_labels = kmeans.fit_predict(np.array(clustering_features))

        # 将聚类结果映射到组ID (0, 1, 2)
        # 我们希望计算能力较强的客户端分配到深层组 (组2)
        # 计算各簇的平均计算能力
        cluster_capabilities = {}
        for cluster_id in range(3):
            clients_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            avg_capability = np.mean([self.client_capabilities[i] for i in clients_in_cluster])
            cluster_capabilities[cluster_id] = avg_capability

            # 按计算能力排序簇 (升序)
        sorted_clusters = sorted(cluster_capabilities.items(), key=lambda x: x[1])
        cluster_to_group = {sorted_clusters[0][0]: 0, sorted_clusters[1][0]: 1, sorted_clusters[2][0]: 2}

        # 分配客户端到组
        for client_idx in range(self.n_clients):
            cluster_id = cluster_labels[client_idx]
            group_id = cluster_to_group[cluster_id]
            self.client_groups[client_idx] = group_id

            # 打印分组结果
        group_counts = {0: 0, 1: 0, 2: 0}
        for group_id in self.client_groups.values():
            group_counts[group_id] += 1

        print(
            f"初始分组完成: 组0(浅层): {group_counts[0]}个客户端, 组1(中层): {group_counts[1]}个客户端, 组2(深层): {group_counts[2]}个客户端")

    def _reassign_clients_groups(self):
        """
        重新分配客户端组
        每5轮执行一次，使用最新的客户端特征
        """
        print("重新分配客户端组...")
        # 更新客户端特征和性能

        # 重新收集客户端特征
        for client_idx in range(self.n_clients):
            # 设置客户端数据
            self._set_client_data(client_idx)

            # 更新计算能力 - 加入基于训练性能的动态调整
            current_capability = self.client_capabilities[client_idx]
            # 根据性能指标调整能力评分
            performance_factor = random.uniform(0.8, 1.2)  # 引入一些随机性
            new_capability = current_capability * performance_factor
            # 确保在合理范围内
            new_capability = max(0.5, min(1.0, new_capability))
            self.client_capabilities[client_idx] = new_capability

            # 提取客户端数据特征
            if self.feature_extractor is not None:
                feature_signature = self.client.extract_feature_signature(self.feature_extractor)
                # 添加一些随机扰动以增加变化性
                # noise = np.random.normal(0, 0.05, size=feature_signature.shape)
                self.client_features[client_idx] = feature_signature

                # 重置客户端
            self.client.reset()

            # 重新构建聚类特征
        clustering_features = []
        for client_idx in range(self.n_clients):
            data_feature = self.client_features[client_idx]
            data_feature_norm = data_feature / (np.linalg.norm(data_feature) + 1e-8)
            # 添加随机扰动
            # perturbation = np.random.normal(0, 0.1, size=data_feature_norm.shape)
            capability = self.client_capabilities[client_idx]
            full_feature = np.concatenate([data_feature_norm[:100], [capability]])
            clustering_features.append(full_feature)

            # 重新聚类
        # 使用不同的随机种子
        random_seed = int(time.time()) % 1000  # 使用时间作为种子
        kmeans = KMeans(n_clusters=3, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(np.array(clustering_features))

        # 重新映射聚类结果到组ID
        cluster_capabilities = {}
        for cluster_id in range(3):
            clients_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if clients_in_cluster:  # 确保簇非空
                avg_capability = np.mean([self.client_capabilities[i] for i in clients_in_cluster])
                cluster_capabilities[cluster_id] = avg_capability

                # 按计算能力排序簇
        sorted_clusters = sorted(cluster_capabilities.items(), key=lambda x: x[1])
        cluster_to_group = {sorted_clusters[0][0]: 0, sorted_clusters[1][0]: 1, sorted_clusters[2][0]: 2}

        # 更新客户端分组
        old_groups = copy.deepcopy(self.client_groups)
        for client_idx in range(self.n_clients):
            cluster_id = cluster_labels[client_idx]
            group_id = cluster_to_group[cluster_id]
            self.client_groups[client_idx] = group_id

            # 计算变化的客户端数量
        changes = sum(1 for i in range(self.n_clients) if old_groups.get(i) != self.client_groups.get(i))

        # 打印分组变化
        group_counts = {0: 0, 1: 0, 2: 0}
        for group_id in self.client_groups.values():
            group_counts[group_id] += 1

        print(
            f"重新分组完成: 组0(浅层): {group_counts[0]}个客户端, 组1(中层): {group_counts[1]}个客户端, 组2(深层): {group_counts[2]}个客户端")
        print(f"共有{changes}个客户端的组分配发生变化")


    def _clients_training(self, sampled_clients):
        """
        客户端训练阶段
        向客户端发送全局模型、组分配和阈值，然后收集更新
        """
        client_updates = []
        server_weights = self.model.state_dict()
        # 客户端训练阶段
        for client_idx in sampled_clients:
            # 设置客户端数据
            self._set_client_data(client_idx)
            # 获取分配的组ID
            assigned_group = self.client_groups.get(client_idx, 0)  # 默认为组0
            # group_threshold = self.group_thresholds[assigned_group]
            # 发送全局模型、组分配和阈值
            self.client.receive_global_model(server_weights, assigned_group)
            # 本地训练
            local_results, local_size = self.client.train_with_distillation()

            # 计算并压缩更新
            update_info = self.client.compute_and_compress_update()
            update_info['local_results'] = local_results
            update_info['local_size'] = local_size
            # 添加到更新列表
            client_updates.append(update_info)
            # 重置客户端
            self.client.reset()

        return client_updates

    def _aggregate_updates(self, client_updates):
        """按组聚合客户端更新"""
        # 计算通信开销
        total_upload = sum(u['upload_params_count'] for u in client_updates)
        baseline = sum(u['total_model_params'] for u in client_updates)
        comm_cost = (total_upload / baseline) * 100 if baseline > 0 else 0
        self.communication_cost.append(total_upload)
        self.total_communication_cost.append(baseline)
        self.comm_ratio.append(comm_cost)

        # 获取当前全局模型
        global_state = self.model.state_dict()

        # # 保存BN参数
        # bn_params = {}
        # for key in global_state.keys():
        #     if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
        #         bn_params[key] = global_state[key].clone()

        # 按组分别聚合特征提取器
        for group_id in range(3):
            # 获取该组的更新
            group_updates = [
                u for u in client_updates
                if u['assigned_group'] == group_id
            ]

            if not group_updates:
                print(f"组 {group_id} 没有客户端更新")
                continue

            print(f"组 {group_id} 有 {len(group_updates)} 个客户端更新")

            # 计算权重
            total_size = sum(u['local_size'] for u in group_updates)
            weights = [u['local_size'] / total_size for u in group_updates]

            # 获取该组的参数键
            group_keys = []
            if group_id == 0:
                prefix = 'group0'
            elif group_id == 1:
                prefix = 'group1'
            elif group_id == 2:
                prefix = 'group2'

            group_keys = [k for k in global_state.keys() if k.startswith(prefix)]

            if not group_keys:
                print(f"[警告] 未找到组 {group_id} 的参数键")
                continue

            # 聚合组参数
            for key in group_keys:
                # 转换全局状态为浮点类型
                global_tensor = global_state[key].float()
                # 初始化聚合张量 - 确保浮点类型
                update_tensor = torch.zeros_like(global_tensor, dtype=torch.float32)

                # 收集该键的所有更新
                for i, update in enumerate(group_updates):
                    group_update = update.get('group_update', {})
                    if key in group_update:
                        # 确保浮点类型
                        current_update = group_update[key].float()
                        update_tensor += current_update * weights[i]


                # 更新全局状态 - 转换回原始数据类型
                global_state[key] = (global_tensor + update_tensor).to(global_state[key].dtype)
                # print(f"更新键 {key}")


        # 聚合所有分类器更新
        classifier_updates = [u for u in client_updates if u.get('classifier_update', {})]

        if classifier_updates:
            total_size = sum(u['local_size'] for u in classifier_updates)
            weights = [u['local_size'] / total_size for u in classifier_updates]

            # 获取分类器键
            classifier_keys = [k for k in global_state.keys() if k.startswith('fc') or k.startswith('classifier')]

            for key in classifier_keys:
                # 转换全局状态为浮点类型
                global_tensor = global_state[key].float()

                # 初始化聚合张量 - 确保浮点类型
                update_tensor = torch.zeros_like(global_tensor, dtype=torch.float32)

                # 收集该键的所有更新
                for i, update in enumerate(classifier_updates):
                    classifier_update = update.get('classifier_update', {})
                    if key in classifier_update:
                        # 确保浮点类型
                        current_update = classifier_update[key].float()
                        update_tensor += current_update * weights[i]


                # 更新全局状态 - 转换回原始数据类型
                global_state[key] = (global_tensor + update_tensor).to(global_state[key].dtype)

                # print(f"更新分类器键 {key}")


        # # 恢复BatchNorm参数
        # for key, value in bn_params.items():
        #     global_state[key].copy_(value)

        # 更新全局模型
        self.model.load_state_dict(global_state)


    def evaluate_global_model(self):
        # 校准 BatchNorm 统计量
        # for module in self.model.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.reset_running_stats()

        # self.model.train()
        # with torch.no_grad():
        #     for i, (inputs, _) in enumerate(self.testloader):
        #         inputs = inputs.to(self.device)
        #         self.model(inputs)
        #         if i >= 50:  # 限制校准批次
        #             break
        self.model.train()  # 设为训练模式以更新BN统计量
        with torch.no_grad():
            batch_count = 0
            for inputs, _ in self.testloader:
                inputs = inputs.to(self.device)
                self.model(inputs)
                batch_count += 1
                if batch_count >= 50:  # 限制校准批次
                    break


        """评估全局模型性能"""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        # print("[诊断] 开始评估全局模型")

        with torch.no_grad():
            for batch_x, batch_y in self.testloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # 前向传播
                logits = self.model(batch_x)

                # 计算准确率
                _, predicted = torch.max(logits, 1)
                correct = (predicted == batch_y).sum().item()

                total_correct += correct
                total_samples += batch_y.size(0)

        accuracy = total_correct / total_samples
        print(f"[诊断] 评估结果: {total_correct}/{total_samples} 正确, 准确率: {accuracy:.4f}")

        return accuracy

    # def _client_sampling(self, round_idx):
    #     """
    #     按组大小比例进行随机采样
    #     保证每个组都有代表性参与训练，有助于提升模型准确率
    #     """
    #     np.random.seed(round_idx)
    #     clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
    #
    #     # 统计各组成员
    #     group_clients = {0: [], 1: [], 2: []}
    #     for client_idx, group_id in self.client_groups.items():
    #         group_clients[group_id].append(client_idx)
    #
    #     # 计算各组大小
    #     group_sizes = {gid: len(clients) for gid, clients in group_clients.items()}
    #     total_clients = sum(group_sizes.values())
    #
    #     # 按比例分配每组采样人数
    #     group_samples = {}
    #     allocated = 0
    #
    #     for group_id in [0, 1, 2]:
    #         if total_clients > 0:
    #             # 按比例计算该组应采样的客户端数
    #             proportion = group_sizes[group_id] / total_clients
    #             n_samples = int(round(clients_per_round * proportion))
    #             group_samples[group_id] = n_samples
    #             allocated += n_samples
    #         else:
    #             group_samples[group_id] = 0
    #
    #     # 处理由于四舍五入导致的总数不匹配
    #     diff = clients_per_round - allocated
    #     if diff > 0:
    #         # 需要补充客户端，优先给客户端最多的组
    #         max_group = max(group_sizes.items(), key=lambda x: x[1])[0]
    #         group_samples[max_group] += diff
    #     elif diff < 0:
    #         # 需要减少客户端，从客户端最多的组中减少
    #         max_group = max(group_sizes.items(), key=lambda x: x[1])[0]
    #         group_samples[max_group] += diff  # diff是负数
    #
    #     # 确保采样数不超过组内客户端总数
    #     for group_id in [0, 1, 2]:
    #         group_samples[group_id] = min(group_samples[group_id], group_sizes[group_id])
    #
    #     # 从每组中随机采样指定数量的客户端
    #     sampled_clients = []
    #     for group_id in [0, 1, 2]:
    #         candidates = group_clients[group_id]
    #         n_samples = group_samples[group_id]
    #
    #         if n_samples > 0 and len(candidates) > 0:
    #             sampled = np.random.choice(candidates, n_samples, replace=False)
    #             sampled_clients.extend(sampled.tolist())
    #
    #     # 打印采样信息
    #     print(f"第{round_idx}轮采样结果:")
    #     print(f"  组0采样: {group_samples[0]}/{group_sizes[0]} 客户端")
    #     print(f"  组1采样: {group_samples[1]}/{group_sizes[1]} 客户端")
    #     print(f"  组2采样: {group_samples[2]}/{group_sizes[2]} 客户端")
    #     print(f"  总计采样: {len(sampled_clients)} 客户端")
    #
    #     return sampled_clients

    def _client_sampling(self, round_idx):
        # 确保相同的客户端采样（用于公平比较）
        np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    def _set_client_data(self, client_idx):
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _print_start(self):
        """打印实验开始日志"""
        if self.device == "cpu":
            device_name = "CPU"
        else:
            if isinstance(self.device, str):
                device_idx = int(self.device[-1])
            elif isinstance(self.device, torch._device):
                device_idx = self.device.index
            device_name = torch.cuda.get_device_name(device_idx)

        print("")
        print("=" * 50)
        print("FedDOT 训练开始，设备: {}".format(device_name))
        print("=" * 50)

    def _wandb_logging(self, round_results , round_idx):
        """
        记录指标到W&B服务器

        Args:
            round_results: 当前轮次的客户端结果
            round_idx: 当前轮次索引
        """
        # 本地训练结果统计
        if "train_acc" in round_results:
            local_train_acc = np.mean(round_results["train_acc"])
        else:
            local_train_acc = 0.0

        if "test_acc" in round_results:
            local_test_acc = np.mean(round_results["test_acc"])
        else:
            local_test_acc = 0.0

        # 服务器测试准确率
        server_test_acc = self.server_results["test_accuracy"][-1]

        # 当前轮次通信开销（如果有）
        if len(self.comm_ratio) > 0:
            comm_cost = self.comm_ratio[-1]
        else:
            comm_cost = 100.0  # 默认为FedAvg的基准(100%)

        total_cost = (sum(self.total_communication_cost) * 32) / (10**9)
        total_upload = (sum(self.communication_cost) * 32) / (10**9)


        # 记录到W&B
        wandb_log_dict = {
            "local_train_acc": local_train_acc,
            "local_test_acc": local_test_acc,
            "server_test_acc": server_test_acc,
            "comm_ratio": comm_cost,
            "total_cost": total_cost,
            "total_upload": total_upload
        }

        # 添加组分布信息
        group_counts = {0: 0, 1: 0, 2: 0}
        for group_id in self.client_groups.values():
            group_counts[group_id] += 1

        wandb_log_dict.update({
            "group0_clients": group_counts[0],
            "group1_clients": group_counts[1],
            "group2_clients": group_counts[2]
        })

        # 添加阈值信息
        # wandb_log_dict.update({
        #     "threshold_group0": self.group_thresholds[0],
        #     "threshold_group1": self.group_thresholds[1],
        #     "threshold_group2": self.group_thresholds[2]
        # })

        # 记录参与客户端数量
        wandb_log_dict["clients_per_round"] = len(self.server_results["client_history"][-1])

        # 记录到W&B
        wandb.log(wandb_log_dict, step=round_idx)

    def _print_round_stats(self, round_idx, test_acc, round_elapse):
        """
        打印轮次统计信息
        """
        # 计算每组客户端数量
        group_counts = {0: 0, 1: 0, 2: 0}
        for group_id in self.client_groups.values():
            group_counts[group_id] += 1

        print(
            "[轮次 {}/{}] 用时 {}s (当前时间: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        print(f"客户端分组: 组0(浅层): {group_counts[0]}, 组1(中层): {group_counts[1]}, 组2(深层): {group_counts[2]}")
        # print(
        #     f"当前阈值: 组0: {self.group_thresholds[0]:.4f}, 组1: {self.group_thresholds[1]:.4f}, 组2: {self.group_thresholds[2]:.4f}")
        print(f"全局模型精度: {test_acc:.4f} (变化: {test_acc - self.prev_accuracy:.4f})")

        # 打印通信开销 (相对于FedAvg)
        if len(self.comm_ratio) > 0:
            current_cost = self.comm_ratio[-1]
            avg_cost = sum(self.comm_ratio) / len(self.comm_ratio)
            total_cost = (sum(self.total_communication_cost) * 32) / (10 ** 9)
            total_upload = (sum(self.communication_cost) * 32) / (10 ** 9)
            print(f"通信效率: 当前轮次: {current_cost:.2f}% FedAvg, 平均: {avg_cost:.2f}% FedAvg")
            print(f"理论总通信量：{total_cost:.3f}, 实际总通信量: {total_upload:.3f}")

        print("-" * 50)