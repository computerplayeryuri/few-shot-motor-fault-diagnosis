import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
import h5py
from torch.utils.data import Dataset
import logging
from datetime import datetime

class ConfigBase:
    def __init__(self):
        # 数据路径
        self.data_root = r"D:\python\data\CRWU\CRWU"
        self.preprocessed_dir = os.path.join(self.data_root, "preprocessed")
        self.checkpoint_root = r"D:\python\data\checkpoints1"
        
        # 模型参数
        self.sample_length = 1024
        self.num_ways = 3
        self.dropout_rate = 0.3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 测试参数
        self.test_episodes = 100
        self.test_batch_size = 8
        self.num_workers = 0
        
        # 日志和结果保存
        self.result_dir = "./test_results1"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化日志
        self.logger = self.setup_logger()

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

class MAMLConfig(ConfigBase):
    def __init__(self):
        self.algorithm = "maml"  # 在调用父类初始化前设置属性
        super().__init__()
        self.num_shots = 10
        self.test_shots = 15
        self.test_inner_steps = 5
        self.inner_lr = 0.05
        self.checkpoint_path = os.path.join(self.checkpoint_root, "maml", "maml_best.pth")

class ReptileConfig(ConfigBase):
    def __init__(self):
        self.algorithm = "reptile"  # 提前设置
        super().__init__()
        self.num_shots = 10
        self.test_shots = 20
        self.inner_lr = 0.2
        self.test_inner_steps = 5 
        self.checkpoint_path = os.path.join(self.checkpoint_root, "reptile", "reptile_best.pth")

class ProtoNetConfig(ConfigBase):
    def __init__(self):
        self.algorithm = "protonet"  # 提前设置
        super().__init__()
        self.num_shots = 5
        self.test_shots = 15
        self.distance_metric = "euclidean"
        self.checkpoint_path = os.path.join(self.checkpoint_root, "protonet", "protonet_best.pth")

class CWRUMetaDataset(Dataset):
    def __init__(self, config, mode="test"):
        self.config = config
        self.mode = mode
        self.data_pools = self._load_data()

    def _load_data(self):
        data_pools = {"healthy": [], "fault": {}}
        
        # 加载健康数据
        with h5py.File(os.path.join(self.config.preprocessed_dir, "normal.h5"), 'r') as hf:
            data_pools["healthy"] = np.concatenate([g["data"][:] for g in hf.values()])
        
        # 加载故障数据
        prefix = "test" if self.mode == "test" else "train"
        for fname in os.listdir(self.config.preprocessed_dir):
            if fname.startswith(prefix) and fname.endswith(".h5"):
                parts = fname[:-3].split("_")
                ftype, diam = parts[1], parts[2]
                with h5py.File(os.path.join(self.config.preprocessed_dir, fname), 'r') as hf:
                    data_pools["fault"][f"{ftype}_{diam}"] = np.concatenate(
                        [g["data"][:] for g in hf.values()]
                    )
        return data_pools

    def __getitem__(self, index):
        available_faults = [k for k, v in self.data_pools["fault"].items() 
                          if len(v) >= 2*(self.config.num_shots + self.config.test_shots)]
        selected_faults = random.sample(available_faults, self.config.num_ways-1)
        
        classes = ["healthy"] + selected_faults
        support_set, query_set = [], []
        
        for label, cls in enumerate(classes):
            samples = self.data_pools["healthy"] if cls == "healthy" else self.data_pools["fault"][cls]
            indices = np.random.choice(len(samples), 
                         self.config.num_shots + self.config.test_shots, 
                         replace=False)
            
            # 数据增强
            support = samples[indices[:self.config.num_shots]].copy()
            support = np.roll(support, shift=np.random.randint(50, 200), axis=1)
            support *= 1 + np.random.uniform(-0.1, 0.1)
            support += np.random.normal(0, 0.005, support.shape)
            
            support_set.append((support, label))
            query_set.append((samples[indices[self.config.num_shots:]], label))
        
        s_data = torch.FloatTensor(np.concatenate([s[0] for s in support_set])).unsqueeze(1)
        q_data = torch.FloatTensor(np.concatenate([q[0] for q in query_set])).unsqueeze(1)
        s_labels = torch.LongTensor(np.concatenate([[s[1]]*self.config.num_shots for s in support_set]))
        q_labels = torch.LongTensor(np.concatenate([[q[1]]*self.config.test_shots for q in query_set]))
        
        return s_data, s_labels, q_data, q_labels

    def __len__(self):
        return 2000

class MetaConvModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1),  
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, config.num_ways)  # 调整全连接层

    def forward(self, x):
        return self.classifier(self.encoder(x))

class ProtoNetModel(nn.Module):
    """ProtoNet专用模型"""
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
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
        self.projection = nn.Linear(256, 128)
        
    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(self.projection(x), dim=1)
    
    def compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        return torch.stack([embeddings[labels == l].mean(0) for l in unique_labels])

    def pairwise_distances(self, x, y, metric='euclidean'):
        """
        计算批量样本与原型之间的距离
        参数:
            x: 查询样本嵌入 [num_queries, embedding_dim]
            y: 原型嵌入 [num_classes, embedding_dim]
            metric: 距离度量方式 ('euclidean' 或 'cosine')
        返回:
            距离矩阵 [num_queries, num_classes]
        """
        if metric == 'euclidean':
            return torch.cdist(x, y, p=2)  # 欧氏距离
        elif metric == 'cosine':
            # 余弦相似度转换为距离: distance = 1 - similarity
            return 1 - F.cosine_similarity(
                x.unsqueeze(1),  # 形状 [num_queries, 1, embedding_dim]
                y.unsqueeze(0),  # 形状 [1, num_classes, embedding_dim]
                dim=2
            )
        else:
            raise ValueError(f"不支持的度量方式: {metric}")
class FunctionalReptileConvModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            # 输入形状: (batch, 1, 1024)
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # -> (64, 512)
            
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # -> (128, 256)
            
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # -> (256, 128)
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, config.num_ways)


    def forward(self, x):
        # 确保通过编码器和分类器
        features = self.encoder(x)  # 编码器输出形状: (batch_size, 128)
        logits = self.classifier(features)  # 分类器输出形状: (batch_size, num_ways)
        return logits
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
            "reptile": FunctionalReptileConvModel,
            "protonet": ProtoNetModel
        }
        return model_map[self.config.algorithm](self.config)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # 提取模型参数
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
        
        # 过滤不匹配的键
        model_dict = self.model.state_dict()
        matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        # 加载匹配参数并保留未匹配部分的初始化
        model_dict.update(matched_dict)
        self.model.load_state_dict(model_dict)
        
        # 确保参数需要梯度
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.config.logger.info(f"参数加载完成，匹配{len(matched_dict)}/{len(model_dict)}个参数")

    def _prepare_test_data(self):
        test_config = self.config.__class__()
        test_config.mode = "test"
        
        # 设置测试模式
        if self.test_mode == "cross_scenario":
            test_config.test_fault_types = ["OR"]
        elif self.test_mode == "cross_diameter":
            test_config.test_diameters = ["028"]
        
        return DataLoader(
            CWRUMetaDataset(test_config, mode="test"),
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

    def _preprocess_input(self, data):
        """关键维度修正"""
        return data.view(-1, 1, self.config.sample_length)

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
        
        # 移除外部的torch.no_grad()，改为按需控制
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
    
    config_class = algorithms[choice][1]()
    test_modes = ["normal", "cross_scenario", "cross_diameter"]
    
    for mode in test_modes:
        config_class.logger.info(f"\n{'='*40}")
        config_class.logger.info(f"开始 {mode} 模式测试")
        
        tester = CWRUTester(config_class, test_mode=mode)
        tester.run_test()
        
        config_class.logger.info(f"{mode} 测试完成\n{'='*40}")