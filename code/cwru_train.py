import os
import re
import random
import logging
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import higher
from datetime import datetime
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import optuna
from optuna.trial import Trial
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import json
from typing import Dict
# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(False)
class ParameterManager:
    @staticmethod
    def save_params(params: Dict, algorithm: str, save_dir: str = "./saved_params"):
        """保存最佳参数到JSON文件"""
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{algorithm}_best_params.json")
        
        try:
            with open(filename, "w") as f:
                json.dump(params, f, indent=4)
            print(f"★ 参数已保存至 {filename}")
        except Exception as e:
            print(f"参数保存失败: {str(e)}")

    @staticmethod
    def load_params(algorithm: str, save_dir: str = "./saved_params") -> Dict:
        """从JSON文件加载参数"""
        filename = os.path.join(save_dir, f"{algorithm}_best_params.json")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"参数文件 {filename} 不存在")
        
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"参数加载失败: {str(e)}")
            return {}
# ----------------------------
# 基础配置类
# ----------------------------
class ConfigBase:
    def __init__(self):
        # 数据路径
        self.data_root = r"D:\python\data\CRWU\CRWU"
        self.preprocessed_dir = os.path.join(self.data_root, "preprocessed")
        
        # 故障类型
        self.train_fault_types = ["IR", "B", "OR"]
        self.test_fault_types = ["IR", "B"]
        self.train_diameters = ['014', '021']
        self.test_diameters = ['028']
        
        # 通用参数
        self.sample_length = 1024
        self.num_ways = 3
        self.dropout_rate = 0.3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 训练参数
        self.num_workers = 0  # Windows下需设为0
        
        # 日志和检查点
        self.save_model = True
        self.save_interval = 5
        self.resume_training = False
        self.algorithm = "base"
        self.log_dir = "./logs1"
        self.checkpoint_dir = "./checkpoints1"
        self.log_level = logging.INFO
        self.logger = self.setup_logger()
        
        # 初始化目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_logger(self):
        logger = logging.getLogger(self.algorithm)
        if logger.handlers:
            return logger
        
        logger.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件日志
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.algorithm}_{datetime.now().strftime('%Y%m%d%H%M')}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

# ----------------------------
# 算法配置类
# ----------------------------
class MAMLConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__()
        self.algorithm = "maml"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "maml")
        
        # 默认参数
        self.num_shots = 10
        self.test_shots = 15
        self.meta_lr = 1e-3
        self.inner_lr = 0.05
        self.num_inner_steps = 5
        self.meta_batch_size = 8
        self.num_epochs = 50
        
        # 如果提供trial对象，则进行参数优化
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.inner_lr = trial.suggest_float('inner_lr', 1e-3, 0.1, log=True)
            self.num_inner_steps = trial.suggest_int('num_inner_steps', 1, 10)
            self.meta_batch_size = trial.suggest_categorical('meta_batch_size', [4, 8, 16])
            self.num_epochs = trial.suggest_int('num_epochs', 20, 100)

class ReptileConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__()
        self.algorithm = "reptile"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "reptile")
        
        # 默认参数
        self.num_shots = 10
        self.test_shots = 20
        self.meta_lr = 2e-4
        self.inner_lr = 0.2
        self.num_inner_steps = 3
        self.meta_batch_size = 8
        self.num_epochs = 300
        self.hidden_dim = 128
        
        # 参数优化
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.inner_lr = trial.suggest_float('inner_lr', 1e-2, 0.5, log=True)
            self.num_inner_steps = trial.suggest_int('num_inner_steps', 1, 5)
            self.meta_batch_size = trial.suggest_categorical('meta_batch_size', [4, 8, 16])
            self.num_epochs = trial.suggest_int('num_epochs', 100, 500)
            self.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])

class ProtoNetConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__()
        self.algorithm = "protonet"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, "protonet")
        
        # 默认参数
        self.num_shots = 5
        self.test_shots = 15
        self.meta_lr = 1e-3
        self.distance_metric = "euclidean"
        self.num_epochs = 50
        self.meta_batch_size = 8
        
        # 参数优化
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.num_shots = trial.suggest_int('num_shots', 1, 10)
            self.num_epochs = trial.suggest_int('num_epochs', 20, 100)
            self.meta_batch_size = trial.suggest_categorical('meta_batch_size', [4, 8, 16])
            self.distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'cosine'])

# ----------------------------
# 数据预处理
# ----------------------------
class CWRUPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        os.makedirs(config.preprocessed_dir, exist_ok=True)

    def _load_mat(self, file_path):
        try:
            mat_data = scio.loadmat(file_path)
            de_key = [k for k in mat_data.keys() if "DE_time" in k][0]
            return mat_data[de_key].flatten()
        except Exception as e:
            print(f"加载失败 {file_path}: {str(e)}")
            return None

    def _process_category(self, category, fault_type, diameter):
        """处理单个故障类别（修复版）"""
        # 构建路径
        if category == "normal":
            data_dir = os.path.join(self.config.data_root, "Normal Baseline")
            h5_name = "normal.h5"
            pattern = re.compile(r"^normal_\d+\.mat$", re.IGNORECASE)
        else:
            sub_dir = {
                "IR": "Inner Race",
                "B": "Ball",
                "OR": "Outer Race"
            }[fault_type]
            dir_diam = f"{diameter} inch"
            data_dir = os.path.join(
                self.config.data_root,
                "12k Fan End Bearing Fault Data",
                sub_dir,
                dir_diam
            )
            h5_name = f"{category}_{fault_type}_{diameter}.h5"
            # 修改后的更灵活的文件名匹配模式
            pattern_str = rf".*{fault_type}.*{diameter}.*\.mat$"
            pattern = re.compile(pattern_str, re.IGNORECASE)

        # 检查目录存在性
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 创建HDF5文件
        h5_path = os.path.join(self.config.preprocessed_dir, h5_name)
        with h5py.File(h5_path, 'w') as hf:
            samples_loaded = 0
            for root, _, files in os.walk(data_dir):
                for file in tqdm(files, desc=f"Processing {h5_name}"):
                    if pattern.match(file):
                        file_path = os.path.join(root, file)
                        data = self._load_mat(file_path)
                        if data is not None:
                            # 确保数据长度一致
                            if len(data) >= self.config.sample_length:
                                data = data[:self.config.sample_length]
                                # 标准化数据
                                data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
                                group = hf.create_group(os.path.splitext(file)[0])
                                group.create_dataset("data", data=data)
                                group.attrs["label"] = 1 if category != "normal" else 0
                                samples_loaded += 1
            
            if samples_loaded == 0:
                raise ValueError(f"未找到匹配文件: {pattern.pattern} in {data_dir}")

    def run(self):
        """执行完整预处理流程"""
        # 处理正常样本
        self._process_category("normal", "", "")
        
        # 处理训练集故障
        for ftype in self.config.train_fault_types:
            for diam in self.config.train_diameters:
                self._process_category("train", ftype, diam)
        
        # 处理测试集故障
        for ftype in self.config.test_fault_types:
            for diam in self.config.test_diameters:
                self._process_category("test", ftype, diam)

# ----------------------------
# 元数据集
# ----------------------------
class CWRUMetaDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.data_pools = self._load_data()
        self.logger = config.logger


    def _load_data(self):
        """加载预处理后的HDF5数据（修复版）"""
        data_pools = {"healthy": [], "fault": {}}
        
        # 加载健康数据（添加异常处理）
        normal_path = os.path.join(self.config.preprocessed_dir, "normal.h5")
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"正常样本文件不存在: {normal_path}")
        
        try:
            with h5py.File(normal_path, 'r') as hf:
                healthy_samples = []
                for group in hf.values():
                    if 'data' in group:
                        healthy_samples.append(group["data"][:])
                if healthy_samples:
                    data_pools["healthy"] = np.concatenate(healthy_samples)
                else:
                    raise ValueError("正常样本文件中没有有效数据")
        except Exception as e:
            raise RuntimeError(f"加载正常数据失败: {str(e)}")

        # 加载故障数据（改进版本）
        prefix = "train" if self.mode == "train" else "test"
        valid_files = [f for f in os.listdir(self.config.preprocessed_dir) 
                      if f.startswith(prefix) and f.endswith(".h5")]
        
        if not valid_files:
            raise FileNotFoundError(f"未找到{prefix}开头的HDF5文件")
        
        for fname in valid_files:
            try:
                parts = fname[:-3].split("_")
                if len(parts) < 3:
                    continue
                    
                ftype, diam = parts[1], parts[2]
                key = f"{ftype}_{diam}"
                h5_path = os.path.join(self.config.preprocessed_dir, fname)
                
                with h5py.File(h5_path, 'r') as hf:
                    fault_samples = []
                    for group in hf.values():
                        if 'data' in group:
                            fault_samples.append(group["data"][:])
                    
                    if fault_samples:
                        data_pools["fault"][key] = np.concatenate(fault_samples)
                    else:
                        self.config.logger.warning(f"文件 {fname} 中没有有效数据")
            except Exception as e:
                self.config.logger.error(f"加载文件 {fname} 失败: {str(e)}")
                continue
        
        # 验证数据是否加载成功
        if not data_pools["healthy"].any():
            raise ValueError("未加载到有效的健康样本数据")
        
        if not data_pools["fault"]:
            raise ValueError("未加载到有效的故障样本数据")
            
        return data_pools

    def __getitem__(self, index):
        # 生成元任务（增强数据增强）
        available_faults = [
            k for k, v in self.data_pools["fault"].items()
            if len(v) >= 2*(self.config.num_shots + self.config.test_shots)
        ]
        num_fault = min(len(available_faults), self.config.num_ways-1)
        selected_faults = random.sample(available_faults, num_fault)
        
        classes = ["healthy"] + selected_faults
        support_set, query_set = [], []
        
        for label, cls in enumerate(classes):
            samples = self.data_pools["healthy"] if cls == "healthy" else self.data_pools["fault"][cls]
            indices = np.random.choice(len(samples), 
                                     self.config.num_shots + self.config.test_shots,
                                     replace=False)
            
            # 增强数据增强
            support = samples[indices[:self.config.num_shots]].copy()
            # 时间偏移
            if random.random() > 0.5:
                shift = np.random.randint(50, 200)
                support = np.roll(support, shift=shift, axis=1)
            # 随机缩放
            scale = 1 + np.random.uniform(-0.1, 0.1)
            support *= scale
            # 添加噪声
            support += np.random.normal(0, 0.005, support.shape)
            
            support_set.append((support, label))
            query_set.append((samples[indices[self.config.num_shots:]], label))
        
        # 合并数据
        s_data = np.concatenate([s[0] for s in support_set])
        s_labels = np.concatenate([[s[1]]*self.config.num_shots for s in support_set])
        q_data = np.concatenate([q[0] for q in query_set])
        q_labels = np.concatenate([[q[1]]*self.config.test_shots for q in query_set])
        
        return (
            torch.FloatTensor(s_data).unsqueeze(1),
            torch.LongTensor(s_labels),
            torch.FloatTensor(q_data).unsqueeze(1),
            torch.LongTensor(q_labels)
        )

    def __len__(self):
        return 2000  # 固定任务数量

# ----------------------------
# 模型定义
# ----------------------------
class ReptileModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, config.num_ways)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

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
        self.classifier = nn.Linear(64, config.num_ways)

    def forward(self, x):
        return self.classifier(self.encoder(x))

class ProtoNetModel(nn.Module):
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
        if metric == 'euclidean':
            return torch.cdist(x, y, p=2)
        elif metric == 'cosine':
            return 1 - F.cosine_similarity(
                x.unsqueeze(1),
                y.unsqueeze(0),
                dim=2
            )
        else:
            raise ValueError(f"不支持的度量方式: {metric}")

# ----------------------------
# 训练工具函数
# ----------------------------
def save_checkpoint(state, config, is_best=False):
    filename = f"{config.algorithm}_"
    if is_best:
        filename += "best.pth"
    else:
        filename += f"epoch{state['epoch']}.pth"
    path = os.path.join(config.checkpoint_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)  
    torch.save(state, path)
    config.logger.info(f"保存检查点到 {path}")

def load_checkpoint(config, model, optimizer=None):
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    config.logger.info(f"加载检查点：epoch {checkpoint['epoch']}, 最佳acc {checkpoint.get('best_acc',0):.2%}")
    return checkpoint

# ----------------------------
# 训练函数
# ----------------------------
def train_maml(model, train_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    best_acc = 0.0
    
    # 训练开始日志
    config.logger.info(
        f"\n{'='*40}\n"
        f"Starting MAML Training\n"
        f"Meta LR: {config.meta_lr:.2e} | Inner LR: {config.inner_lr:.2e}\n"
        f"Inner Steps: {config.num_inner_steps} | Batch Size: {config.meta_batch_size}\n"
        f"Epochs: {config.num_epochs}\n"
        f"{'='*40}"
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss, total_acc, batch_count = 0.0, 0.0, 0
        
        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)
            
            optimizer.zero_grad()
            meta_loss = 0.0
            task_accs = []
            
            with higher.innerloop_ctx(model, optimizer, 
                                    copy_initial_weights=False,
                                    track_higher_grads=True) as (fmodel, diffopt):
                # 内循环适应
                for task_idx in range(config.meta_batch_size):
                    for step in range(config.num_inner_steps):
                        s_out = fmodel(s_d[task_idx])
                        s_loss = F.cross_entropy(s_out, s_l[task_idx])
                        diffopt.step(s_loss)
                    
                    # 查询集评估
                    q_out = fmodel(q_d[task_idx])
                    q_loss = F.cross_entropy(q_out, q_l[task_idx])
                    meta_loss += q_loss / config.meta_batch_size
                    
                    # 计算准确率
                    preds = q_out.argmax(dim=1)
                    acc = (preds == q_l[task_idx]).float().mean()
                    task_accs.append(acc.item())
                
                # 元优化
                meta_loss.backward()
                optimizer.step()
            
            # 累计统计
            batch_loss = meta_loss.item()
            batch_acc = np.mean(task_accs)
            total_loss += batch_loss
            total_acc += batch_acc
            batch_count += 1
            
            # 每5个batch记录一次
            if (batch_idx + 1) % 5 == 0:
                config.logger.info(
                    f"Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2%}"
                )
        
        # Epoch统计
        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count
        
        # 保存检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': vars(config)
            }, config)
        
        # 更新最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': vars(config)
            }, config, is_best=True)
            config.logger.info(f"★ 发现新最佳模型，准确率: {best_acc:.2%}")
        
        # Epoch总结日志
        config.logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}] Summary | "
            f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%} | "
            f"Best Acc: {best_acc:.2%}\n"
            f"{'-'*60}"
        )
    
    # 训练结束日志
    config.logger.info(
        f"\n{'='*40}\n"
        f"Training Completed | Final Best Acc: {best_acc:.2%}\n"
        f"{'='*40}"
    )
    
    return best_acc

def train_reptile(model, train_loader, config, device):
    model.to(device)
    best_acc = 0.0

    config.logger.info(
        f"\n{'='*40}\n"
        f"Starting Reptile Training\n"
        f"Meta LR: {config.meta_lr:.2e} | Inner LR: {config.inner_lr:.2e}\n"
        f"Inner Steps: {config.num_inner_steps} | Batch Size: {config.meta_batch_size}\n"
        f"Epochs: {config.num_epochs}\n"
        f"{'='*40}"
    )

    for epoch in range(config.num_epochs):
        model.train()
        total_loss, total_acc, batch_count = 0.0, 0.0, 0

        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)

            # 存储原始模型参数
            original_params = {n: p.clone() for n, p in model.named_parameters()}

            batch_loss, batch_acc = 0.0, 0.0

            for task_idx in range(config.meta_batch_size):
                optimizer = optim.SGD(model.parameters(), lr=config.inner_lr)

                # 使用 Higher 进行临时任务内更新
                with higher.innerloop_ctx(
                    model, optimizer,
                    copy_initial_weights=True,
                    track_higher_grads=False  # Reptile 不需要高阶梯度
                ) as (fmodel, diffopt):
                    # Inner loop: 使用 support set 进行快速学习
                    for _ in range(config.num_inner_steps):
                        logits = fmodel(s_d[task_idx])
                        loss = F.cross_entropy(logits, s_l[task_idx])
                        diffopt.step(loss)

                    # 评估 query set 上的准确率
                    q_logits = fmodel(q_d[task_idx])
                    q_loss = F.cross_entropy(q_logits, q_l[task_idx])
                    acc = (q_logits.argmax(1) == q_l[task_idx]).float().mean()

                    batch_loss += q_loss.item() / config.meta_batch_size
                    batch_acc += acc.item() / config.meta_batch_size

                    # 手动更新模型参数（Reptile方式：朝任务参数方向走一步）
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            p.data = p.data + config.meta_lr * (
                                fmodel.named_parameters()[name].data - p.data
                            )

            total_loss += batch_loss
            total_acc += batch_acc
            batch_count += 1

            if (batch_idx + 1) % 5 == 0:
                config.logger.info(
                    f"Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2%}"
                )

        # Epoch统计
        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count

        # 保存模型检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': None,
                'best_acc': best_acc,
                'config': vars(config)
            }, config)

        # 更新最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': None,
                'best_acc': best_acc,
                'config': vars(config)
            }, config, is_best=True)
            config.logger.info(f"★ 发现新最佳模型，准确率: {best_acc:.2%}")

        config.logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}] Summary | "
            f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%} | "
            f"Best Acc: {best_acc:.2%}\n"
            f"{'-'*60}"
        )

    config.logger.info(
        f"\n{'='*40}\n"
        f"Training Completed | Final Best Acc: {best_acc:.2%}\n"
        f"{'='*40}"
    )

    return best_acc

def train_protonet(model, train_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    best_acc = 0.0
    
    # 训练开始日志
    config.logger.info(
        f"\n{'='*40}\n"
        f"Starting ProtoNet Training\n"
        f"Meta LR: {config.meta_lr:.2e} | Distance Metric: {config.distance_metric}\n"
        f"Shots: {config.num_shots} | Batch Size: {config.meta_batch_size}\n"
        f"Epochs: {config.num_epochs}\n"
        f"{'='*40}"
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss, total_acc, batch_count = 0.0, 0.0, 0
        
        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)
            
            optimizer.zero_grad()
            batch_loss, batch_acc = 0.0, 0.0
            
            for task_idx in range(config.meta_batch_size):
                # 处理单个任务
                support = s_d[task_idx]
                support_labels = s_l[task_idx]
                query = q_d[task_idx]
                query_labels = q_l[task_idx]
                
                # 计算嵌入
                support_embeddings = model(support)
                query_embeddings = model(query)
                
                # 计算原型
                prototypes = model.compute_prototypes(support_embeddings, support_labels)
                
                # 计算距离
                distances = model.pairwise_distances(query_embeddings, prototypes, config.distance_metric)
                
                # 计算损失和准确率
                loss = F.cross_entropy(-distances, query_labels)
                preds = torch.argmin(distances, dim=1) if config.distance_metric == 'euclidean' else torch.argmax(-distances, dim=1)
                acc = (preds == query_labels).float().mean()
                
                batch_loss += loss / config.meta_batch_size
                batch_acc += acc / config.meta_batch_size
            
            # 反向传播
            batch_loss.backward()
            optimizer.step()
            
            # 累计统计
            total_loss += batch_loss.item()
            total_acc += batch_acc.item()
            batch_count += 1
            
            # 每5个batch记录一次
            if (batch_idx + 1) % 5 == 0:
                config.logger.info(
                    f"Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {batch_loss.item():.4f} | Acc: {batch_acc.item():.2%}"
                )
        
        # Epoch统计
        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count
        
        # 保存检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': vars(config)
            }, config)
        
        # 更新最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': vars(config)
            }, config, is_best=True)
            config.logger.info(f"★ 发现新最佳模型，准确率: {best_acc:.2%}")
        
        # Epoch总结日志
        config.logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}] Summary | "
            f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%} | "
            f"Best Acc: {best_acc:.2%}\n"
            f"{'-'*60}"
        )
    
    # 训练结束日志
    config.logger.info(
        f"\n{'='*40}\n"
        f"Training Completed | Final Best Acc: {best_acc:.2%}\n"
        f"{'='*40}"
    )
    
    return best_acc

# ----------------------------
# 自动调参训练器
# ----------------------------
class AutoMetaTrainer:
    def __init__(self, algorithm="maml"):
        self.algorithm = algorithm.lower()
        self.best_params = None
        self.best_score = 0
        
    def objective(self, trial):
        """Optuna优化的完整目标函数"""
        # 根据算法选择配置
        if self.algorithm == "maml":
            config = MAMLConfig(trial)
            model = MetaConvModel(config).to(config.device)
            train_func = train_maml
        elif self.algorithm == "reptile":
            config = ReptileConfig(trial)
            model = ReptileModel(config).to(config.device)
            train_func = train_reptile
        elif self.algorithm == "protonet":
            config = ProtoNetConfig(trial)
            model = ProtoNetModel(config).to(config.device)
            train_func = train_protonet
        
        # 数据加载
        train_dataset = CWRUMetaDataset(config, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.meta_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True
        )
        
        # 训练模型
        best_acc = 0.0
        best_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 3
        
        for epoch in range(config.num_epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            model.train()
            
            for batch in train_loader:
                if self.algorithm == "maml":
                    batch_loss, batch_acc = self._maml_train_step(model, batch, config, device=config.device)
                elif self.algorithm == "reptile":
                    batch_loss, batch_acc = self._reptile_train_step(model, batch, config, device=config.device)
                elif self.algorithm == "protonet":
                    batch_loss, batch_acc = self._protonet_train_step(model, batch, config, device=config.device)
                
                epoch_loss += batch_loss / len(train_loader)
                epoch_acc += batch_acc / len(train_loader)
            
            # 报告给Optuna
            trial.report(epoch_acc, epoch)
            
            # 早停检查
            if trial.should_prune():
                config.logger.info(f"Early stopping at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()
            
            # 更新最佳记录
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    break
        
        # 更新全局最佳参数
        if best_acc > self.best_score:
            self.best_score = best_acc
            self.best_params = {
                'num_shots': config.num_shots,
                'test_shots': config.test_shots,
                'meta_lr': config.meta_lr,
                'inner_lr': getattr(config, 'inner_lr', None),
                'num_inner_steps': getattr(config, 'num_inner_steps', None),
                'meta_batch_size': config.meta_batch_size,
                'num_epochs': config.num_epochs,
                'distance_metric': getattr(config, 'distance_metric', None),
                'hidden_dim': getattr(config, 'hidden_dim', None)
            }
        
        # 返回复合指标
        return best_acc * 0.9 + (1 / (best_loss + 1e-8)) * 0.1
    def optimize(self, n_trials=20, save_params=True):
        """执行参数优化并自动保存"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(self.objective, n_trials=n_trials)

        # 记录最佳结果
        self.best_params = study.best_params
        self.best_value = study.best_value

        # 自动保存参数
        if save_params:
            ParameterManager.save_params(
                params={
                    "best_params": self.best_params,
                    "best_value": self.best_value,
                    "algorithm": self.algorithm
                },
                algorithm=self.algorithm
            )
        
        return self.best_params    
    def _maml_train_step(self, model, batch, config, device):
        s_d, s_l, q_d, q_l = [t.to(device) for t in batch]
        optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
        
        optimizer.zero_grad()
        meta_loss = 0.0
        task_accs = []
        
        with higher.innerloop_ctx(model, optimizer, 
                                copy_initial_weights=False,
                                track_higher_grads=True) as (fmodel, diffopt):
            for task_idx in range(config.meta_batch_size):
                # 内循环适应
                for _ in range(config.num_inner_steps):
                    s_out = fmodel(s_d[task_idx])
                    s_loss = F.cross_entropy(s_out, s_l[task_idx])
                    diffopt.step(s_loss)
                
                # 查询集评估
                q_out = fmodel(q_d[task_idx])
                q_loss = F.cross_entropy(q_out, q_l[task_idx])
                meta_loss += q_loss / config.meta_batch_size
                
                # 计算准确率
                preds = q_out.argmax(dim=1)
                acc = (preds == q_l[task_idx]).float().mean()
                task_accs.append(acc.item())
            
            meta_loss.backward()
            optimizer.step()
        
        return meta_loss.item(), np.mean(task_accs)
    
    def _reptile_train_step(self, model, batch, config, device):
        s_d, s_l, q_d, q_l = [t.to(device) for t in batch]
        optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
        
        meta_grads = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}
        batch_loss, batch_acc = 0.0, 0.0
        
        for task_idx in range(config.meta_batch_size):
            with torch.no_grad():
                with higher.innerloop_ctx(model, optimizer, 
                                      copy_initial_weights=True,
                                      track_higher_grads=False) as (fmodel, diffopt):
                    # 适应阶段
                    for _ in range(config.num_inner_steps):
                        outputs = fmodel(s_d[task_idx])
                        loss = F.cross_entropy(outputs, s_l[task_idx])
                        diffopt.step(loss)
                    
                    # 计算元梯度
                    for (n, p), init_p in zip(fmodel.named_parameters(), model.parameters()):
                        delta = (p.detach() - init_p.detach()) / config.meta_batch_size
                        meta_grads[n] += delta
                    
                    # 评估查询集
                    q_outputs = fmodel(q_d[task_idx])
                    loss = F.cross_entropy(q_outputs, q_l[task_idx])
                    acc = (q_outputs.argmax(1) == q_l[task_idx]).float().mean()
                    
                    batch_loss += loss.item() / config.meta_batch_size
                    batch_acc += acc.item() / config.meta_batch_size
        
        # 应用元梯度
        optimizer.zero_grad()
        for n, p in model.named_parameters():
            p.grad = -meta_grads[n].to(p.device)
        optimizer.step()
        
        return batch_loss, batch_acc
    
    def _protonet_train_step(self, model, batch, config, device):
        s_d, s_l, q_d, q_l = [t.to(device) for t in batch]
        optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
        
        optimizer.zero_grad()
        batch_loss, batch_acc = 0.0, 0.0
        
        for task_idx in range(config.meta_batch_size):
            # 处理单个任务
            support = s_d[task_idx]
            support_labels = s_l[task_idx]
            query = q_d[task_idx]
            query_labels = q_l[task_idx]
            
            # 计算嵌入
            support_embeddings = model(support)
            query_embeddings = model(query)
            
            # 计算原型
            prototypes = model.compute_prototypes(support_embeddings, support_labels)
            
            # 计算距离
            distances = model.pairwise_distances(query_embeddings, prototypes, config.distance_metric)
            
            # 计算损失和准确率
            loss = F.cross_entropy(-distances, query_labels)
            preds = torch.argmin(distances, dim=1) if config.distance_metric == 'euclidean' else torch.argmax(-distances, dim=1)
            acc = (preds == query_labels).float().mean()
            
            batch_loss += loss / config.meta_batch_size
            batch_acc += acc / config.meta_batch_size
        
        batch_loss.backward()
        optimizer.step()
        
        return batch_loss.item(), batch_acc.item()

    def optimize(self, n_trials=20):
        """执行参数优化"""
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"\n{'='*40}")
        print(f"最佳参数 ({self.algorithm}):")
        for k, v in self.best_params.items():
            if v is not None:
                print(f"{k}: {v}")
        print(f"最佳验证准确率: {self.best_score:.2%}")
        print('='*40)
        
        return self.best_params

# ----------------------------
# 主程序
# ----------------------------
def main():
    # 确保数据已预处理
    base_config = ConfigBase()
    if not os.path.exists(base_config.preprocessed_dir):
        print("启动数据预处理...")
        try:
            preprocessor = CWRUPreprocessor(base_config)
            preprocessor.run()
            print("数据预处理完成！")
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            return
    
    # 验证预处理数据
    print("验证预处理数据...")
    try:
        # 检查至少有一个训练文件
        train_files = [f for f in os.listdir(base_config.preprocessed_dir) 
                      if f.startswith("train") and f.endswith(".h5")]
        if not train_files:
            raise ValueError("未找到训练数据文件")
            
        # 检查正常样本文件
        if not os.path.exists(os.path.join(base_config.preprocessed_dir, "normal.h5")):
            raise FileNotFoundError("缺少正常样本数据")
            
        print("数据验证通过！")
    except Exception as e:
        print(f"数据验证失败: {str(e)}")
        return
    
    # 一键训练所有模型
    algorithms = ["maml","reptile", "protonet"]
    
    for algo in algorithms:
        print(f"\n{'='*40}")
        print(f"开始训练 {algo.upper()} 模型")
        print('='*40)
        
        # 自动调参
        trainer = AutoMetaTrainer(algorithm=algo)
        best_params = trainer.optimize(n_trials=10)
        
        # 用最佳参数训练最终模型
        if algo == "maml":
            config = MAMLConfig()
            model = MetaConvModel(config).to(config.device)
            train_func = train_maml
        elif algo == "reptile":
            config = ReptileConfig()
            model = ReptileModel(config).to(config.device)
            train_func = train_reptile
        else:  # protonet
            config = ProtoNetConfig()
            model = ProtoNetModel(config).to(config.device)
            train_func = train_protonet
        
        # 应用最佳参数
        for k, v in best_params.items():
            if hasattr(config, k) and v is not None:
                setattr(config, k, v)
        
        # 加载数据集
        train_dataset = CWRUMetaDataset(config, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.meta_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True
        )
        
        # 训练最终模型
        final_acc = train_func(model, train_loader, config, config.device)
        print(f"{algo.upper()} 最终模型准确率: {final_acc:.2%}")

if __name__ == "__main__":
    main()