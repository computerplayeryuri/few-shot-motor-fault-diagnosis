import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import higher
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
# ----------------------------
# 基础配置类
# ----------------------------
class ConfigBase:
    def __init__(self):
        # 数据集配置
        self.data_dir = r"D:\python\data\1\1"
        self.healthy_class_name = "Healthy"
        self.train_faults = [0.7, 1.1]
        self.test_faults = [ 1.3, 1.7]
        self.mode ="train"
        # 通用模型参数
        self.num_ways = 3
        self.sample_length = 1024
        self.dropout_rate = 0.4
        self.algorithm = "reptile"  
        # 通用训练参数
        self.meta_batch_size = 8
        self.num_epochs = 80
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_episodes = 100
        self.test_batch_size = 8
        self.num_workers = 0
        # 模型保存
        self.save_model = True
        self.save_interval = 5
        self.checkpoint_dir = r"D:\python\data\4.28"
        self.resume_training = False
        self.checkpoint_path = ""
        self.test_after_load = True
        self.result_dir = "./test_results430"
        os.makedirs(self.result_dir, exist_ok=True)
        # 硬件相关
        self.num_workers = 4
        self.pin_memory = True
        self.logger = self.setup_logger()
    def algorithm_name(self):
        return self.__class__.__name__.replace('Config', '').lower()
        

    def setup_logger(self):
        logger = logging.getLogger(self.algorithm)
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件日志
        log_file = os.path.join(self.result_dir, f"{self.algorithm}_test.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
# ----------------------------

# ----------------------------
class MAMLConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.algorithm = "maml"
        self.num_shots = 1
        self.test_shots = 15
        self.meta_lr = 3e-3
        self.inner_lr = 0.1
        self.num_inner_steps = 1
        self.embedding_dim = 128
        self.test_inner_steps = 1 
        self.distance_metric = "euclidean"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "maml")

class ReptileConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.algorithm = "reptile"
        self.num_shots = 1
        self.test_shots = 15
        # 优化后的训练参数
        self.meta_lr = 1e-3        # 降低元学习率
        self.inner_lr = 0.05       # 调整内部学习率
        self.num_epochs = 120      # 增加训练轮次
        self.num_inner_steps = 3   # 增加内部更新步数
        self.grad_clip = 1.0       # 放松梯度裁剪
        self.weight_decay = 1e-5   # 添加L2正则化
        self.embedding_dim = 128
        self.test_inner_steps = 3 
        self.distance_metric = "euclidean"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "reptile")

class ProtoNetConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.algorithm = "protonet"
        self.embedding_dim = 32
        self.num_shots = 1
        self.test_shots = 5
        self.meta_lr = 1e-4
        self.distance_metric = "euclidean"  # euclidean 或 cosine
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "protonet")

# ----------------------------
# 增强型数据集类（保持不变）
# ----------------------------
class EnhancedMotorMetaDataset(Dataset):
    def __init__(self, config,mode="train"):
        self.config = config
        self.class_data = {}
        self.valid_classes = []
        
        # 加载数据
        fault_folders = []
        healthy_path = None
        
        # 遍历数据目录
        for folder in os.listdir(config.data_dir):
            full_path = os.path.join(config.data_dir, folder)
            
            if config.healthy_class_name in folder:
                healthy_path = full_path
                continue
                
            match = re.match(r"(\d+\.\d+)mm-bearing-faults", folder)
            if match:
                fault_size = float(match.group(1))
                fault_folders.append((fault_size, full_path))
        
        # 加载健康样本
        if not healthy_path:
            raise ValueError("未找到健康样本文件夹")
        self._load_class_data(0, healthy_path)
        self.valid_classes.append(0)
        
        # 加载故障样本
        target_faults = config.train_faults if config.mode == "train" else config.test_faults
        selected_faults = [(size, path) for size, path in fault_folders if size in target_faults]
        
        for idx, (size, path) in enumerate(selected_faults, 1):
            self._load_class_data(idx, path)
            self.valid_classes.append(idx)
        
        if len(self.valid_classes) < config.num_ways:
            raise ValueError(f"可用类别不足，需要至少 {config.num_ways} 个类别")
        
        self.num_tasks = 2000

    def _load_class_data(self, class_id, folder_path):
        samples = []
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
                
            try:
                df = pd.read_csv(os.path.join(folder_path, file), sep=None, engine="python")
                df.columns = df.columns.str.strip().str.lower()
                
                vib_cols = {}
                for axis in ['x', 'y', 'z']:
                    matches = [col for col in df.columns if re.search(axis, col, re.IGNORECASE)]
                    if not matches:
                        raise ValueError(f"缺失{axis}轴数据: {file}")
                    vib_cols[axis] = matches[0]
                
                data = df[list(vib_cols.values())].values.astype(np.float32)
                data = self._clean_data(data)
                samples.extend(self._split_samples(data))
                
            except Exception as e:
                print(f"加载文件 {file} 失败: {str(e)}")
                continue
        
        if len(samples) >= (self.config.num_shots + self.config.test_shots):
            self.class_data[class_id] = np.stack(samples)
            print(f"类别 {class_id} 加载完成，样本数: {len(samples)}")
        else:
            print(f"跳过类别 {class_id}，样本不足")

    def _clean_data(self, data):
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        valid_mask = (np.abs(data) < 5).all(axis=1)
        return data[valid_mask]

    def _split_samples(self, data):
        return [data[i:i+self.config.sample_length] 
                for i in range(0, len(data)-self.config.sample_length+1, 512)]

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        selected_classes = random.sample(self.valid_classes, self.config.num_ways)
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for task_label, class_id in enumerate(selected_classes):
            samples = self.class_data[class_id]
            indices = np.random.choice(len(samples), 
                                      self.config.num_shots + self.config.test_shots,
                                      replace=False)
            
            # 数据增强
            support = samples[indices[:self.config.num_shots]].copy()
            if random.random() > 0.5:
                support = np.flip(support, axis=1).copy()  # 时间翻转
            if random.random() > 0.5:
                support += np.random.normal(0, 0.05, support.shape)  # 高斯噪声
            
            support_data.append(support)
            support_labels.extend([task_label]*self.config.num_shots)
            
            query = samples[indices[self.config.num_shots:]]
            query_data.append(query)
            query_labels.extend([task_label]*self.config.test_shots)
        
        return (
            torch.as_tensor(np.concatenate(support_data), dtype=torch.float32),
            torch.as_tensor(support_labels, dtype=torch.long),
            torch.as_tensor(np.concatenate(query_data), dtype=torch.float32),
            torch.as_tensor(query_labels, dtype=torch.long)
)
class ProtoNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 投影层
        self.projection = nn.Linear(256, config.embedding_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, L]
        features = self.feature_extractor(x)
        return F.normalize(self.projection(features), p=2, dim=1)

    def compute_prototypes(self, support_embeddings, support_labels):
        """计算每个类别的原型（均值）"""
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]
            prototypes.append(class_embeddings.mean(dim=0))
        return torch.stack(prototypes)

    def pairwise_distances(self, x, y, metric='euclidean'):
        """计算两组嵌入向量之间的成对距离"""
        if metric == 'euclidean':
            return torch.cdist(x, y)
        elif metric == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return 1 - torch.mm(x_norm, y_norm.T)
        else:
            raise ValueError(f"不支持的度量方式: {metric}")    
class ReptileModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取层 (保持不变)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 投影层
        self.projection = nn.Linear(256, config.embedding_dim)

    def forward(self, x):
        # 添加维度置换 (修复通道位置问题)
        x = x.permute(0, 2, 1)  # [B, C, L]
        features = self.feature_extractor(x)
        return F.normalize(self.projection(features), p=2, dim=1)
class MetaConvModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取层保持不变
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(256, config.embedding_dim)
        
    def forward(self, x):
        # 直接输入，无需维度置换
        features = self.feature_extractor(x)
        return F.normalize(self.classifier(features), p=2, dim=1)

class CWRUTester:
    def __init__(self, config, test_mode="normal"):
        self.config = config
        self.test_mode = test_mode
        self.device = torch.device(config.device)
        
        # 初始化模型
        self.model = self._init_model().to(self.device)
        self._load_checkpoint()
        
        # 准备测试数据
        self.test_loader = self._prepare_test_data()

    def _init_model(self):
        model_map = {
            "maml": MetaConvModel,
            "reptile": ReptileModel,
            "protonet": ProtoNetModel
        }
        return model_map[self.config.algorithm](self.config)
    def _load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state', checkpoint)
        
        # 转换投影层键名为分类器
        state_dict = {k.replace('projection', 'classifier'): v 
                     for k, v in state_dict.items()}
        
        # 过滤不匹配的键
        model_dict = self.model.state_dict()
        matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        model_dict.update(matched_dict)
        self.model.load_state_dict(model_dict)
        
        # 确保参数需要梯度
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.config.logger.info(f"检查点参数键:", state_dict.keys())
        print("当前模型键:", model_dict.keys())
        self.config.logger.info(f"参数加载完成，匹配{len(matched_dict)}/{len(model_dict)}个参数")

    def _prepare_test_data(self):
        test_config = self.config.__class__()
        
        # 设置测试模式
        if self.test_mode == "cross_scenario":
            test_config.test_fault_types = ["OR"]
        elif self.test_mode == "cross_diameter":
            test_config.test_diameters = ["028"]
        
        return DataLoader(
            EnhancedMotorMetaDataset(test_config, mode="test"),
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

    def _preprocess_input(self, data):
        """将数据调整为 (batch_size * num_samples, 通道数, 序列长度) """
        # 假设输入形状为 [batch, num_samples, length, channels]
        # 转换为 [batch*num_samples, channels, length]
        data = data.permute(0, 1, 3, 2)  # 调整维度顺序
        return data.reshape(-1, data.size(2), data.size(3))  # 调整为 (B*N, C, L)

    def _protonet_inference(self, s_data, s_labels, q_data):
        s_flat = self._preprocess_input(s_data)
        q_flat = self._preprocess_input(q_data)
        
        s_emb = self.model(s_flat)
        q_emb = self.model(q_flat)
        
        batch_size = s_data.size(0)
        s_emb = s_emb.view(batch_size, -1, s_emb.size(-1))
        q_emb = q_emb.view(batch_size, -1, q_emb.size(-1))
        
        preds = []
        for i in range(batch_size):
            prototypes = self.model.compute_prototypes(s_emb[i], s_labels[i])
            distances = self.model.pairwise_distances(q_emb[i], prototypes)
            preds.append(distances.argmin(dim=1))
        return torch.cat(preds)

    def _maml_reptile_inference(self, s_data, s_labels, q_data):
        self.model.train()
        batch_size = s_data.size(0)
        all_preds = []
        
        s_flat = self._preprocess_input(s_data)
        s_labels_flat = s_labels.view(-1)
        
        # 保存原始参数，避免污染模型状态
        original_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        
        for _ in range(self.config.test_inner_steps):
            # 前向传播（启用梯度）
            outputs = self.model(s_flat)
            loss = F.cross_entropy(outputs, s_labels_flat)
            
            # 计算梯度（create_graph=True允许高阶导数）
            gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            
            # 手动更新参数（禁用梯度追踪）
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), gradients):
                    if grad is not None:
                        param.data.sub_(self.config.inner_lr * grad)
        
        # 查询集预测（禁用梯度）
        with torch.no_grad():
            q_flat = self._preprocess_input(q_data)
            logits = self.model(q_flat)
            preds = logits.argmax(dim=1)
        
        # 恢复原始参数
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.data.copy_(original_params[n].data)
        self.model.eval()
        return preds

    def _apply_gradient(self, gradients, alpha):
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradients):
                param.data.sub_(alpha * grad)

    def run_test(self):
        self.model.eval()
        all_preds, all_labels = [], []
        
        # 移格外部的torch.no_grad()，改为按需控制
        for s_data, s_labels, q_data, q_labels in tqdm(self.test_loader):
            s_data, q_data = s_data.to(self.device), q_data.to(self.device)
            s_labels, q_labels = s_labels.to(self.device), q_labels.to(self.device)
            
            if self.config.algorithm == "protonet":
                # ProtoNet不需要梯度
                with torch.no_grad():
                    preds = self._protonet_inference(s_data, s_labels, q_data)
            else:
                # MAML和Reptile需要梯度更新
                preds = self._maml_reptile_inference(s_data, s_labels, q_data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(q_labels.view(-1).cpu().numpy())
        
        self._save_results(all_labels, all_preds)

    def _save_results(self, true_labels, pred_labels):
        # 生成结果目录
        result_dir = os.path.join(self.config.result_dir, self.config.algorithm, self.test_mode)
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存指标
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        report = classification_report(true_labels, pred_labels, zero_division=0)
        
        with open(os.path.join(result_dir, "results.txt"), 'w') as f:
            f.write(f"Test Mode: {self.test_mode}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}")
        
        # 保存混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
        plt.close()

if __name__ == "__main__":
    # 选择测试配置
    algorithms = {
        "1": ("maml", MAMLConfig),
        "2": ("protonet", ProtoNetConfig),
        "3": ("reptile", ReptileConfig)
    }
    
    print("请选择测试算法:")
    for k, v in algorithms.items():
        print(f"{k}. {v[0]}")
    choice = input("输入选项 (1-3): ")
    
    # 主程序需要添加检查点路径配置
    config_class = algorithms[choice][1]()
    # 新增检查点路径配置
    config_class.checkpoint_path = os.path.join(config_class.checkpoint_dir, f"{config_class.algorithm}_best.pth")  # 需要添加这行
    test_modes = ["normal", "cross_scenario", "cross_diameter"]
    
    for mode in test_modes:
        config_class.logger.info(f"\n{'='*40}")
        config_class.logger.info(f"开始 {mode} 模式测试")
        
        tester = CWRUTester(config_class, test_mode=mode)
        tester.run_test()
        
        config_class.logger.info(f"{mode} 测试完成\n{'='*40}")