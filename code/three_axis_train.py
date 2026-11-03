import os
import re
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import higher
from datetime import datetime
import optuna
from optuna.trial import Trial

# ----------------------------
# 基础配置类（增加日志和调参支持）
# ----------------------------
class ConfigBase:
    def __init__(self, trial=None):
        # 数据集配置
        self.data_dir = r"D:\python\data\1\1"
        self.healthy_class_name = "Healthy"
        self.train_faults = [0.7, 1.1, 1.5]
        self.test_faults = [0.9, 1.3, 1.7]
        self.mode = "train"
        
        # 通用模型参数
        self.num_ways = 4
        self.sample_length = 1024
        self.dropout_rate = 0.4
        self.algorithm = "base"
        
        # 训练参数
        self.meta_batch_size = 8
        self.num_epochs = 80
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 日志和保存
        self.save_model = True
        self.save_interval = 5
        self.checkpoint_dir = os.path.join(r"D:\python\data\1\2", self.algorithm)
        self.resume_training = False
        self.test_after_load = True
        self.num_workers = 4
        self.pin_memory = True
        
        # 日志配置
        self.log_dir = os.path.join(r"D:\python\data\1", "logs")
        self.log_level = logging.INFO
        self.logger = self.setup_logger()
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

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
            nn.Conv1d(3, 32, 5, padding=2), 
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
        self.device = config.device
    def forward(self, x):  
        x = x.to(self.device) 
        x = x.permute(0, 2, 1)  
        features = self.encoder(x)
        return self.classifier(features)
class ProtoNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.projection = nn.Linear(256, config.embedding_dim)
        
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
# 算法配置类（支持Optuna调参）
# ----------------------------
class MAMLConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__(trial)
        self.algorithm = "maml"
        self.checkpoint_dir = os.path.join(r"D:\python\data\1", "maml")
        
        # 默认参数
        self.num_shots = 1
        self.test_shots = 15
        self.meta_lr = 3e-3
        self.inner_lr = 0.1
        self.num_inner_steps = 1
        
        # Optuna参数搜索空间
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.inner_lr = trial.suggest_float('inner_lr', 1e-3, 0.5, log=True)
            self.num_inner_steps = trial.suggest_int('num_inner_steps', 1, 5)
            self.meta_batch_size = trial.suggest_categorical('meta_batch_size', [4, 8, 16])
            self.num_epochs = trial.suggest_int('num_epochs', 20, 100)

class ReptileConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__(trial)
        self.algorithm = "reptile"
        self.checkpoint_dir = os.path.join(r"D:\python\data\1", "reptile")
        
        # 默认参数
        self.num_shots = 1
        self.test_shots = 15
        self.meta_lr = 3e-3
        self.inner_lr = 0.15
        self.num_inner_steps = 8
        self.momentum = 0.85
        self.weight_decay = 1e-4
        
        # Optuna参数
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.inner_lr = trial.suggest_float('inner_lr', 1e-2, 0.5, log=True)
            self.num_inner_steps = trial.suggest_int('num_inner_steps', 1, 10)
            self.meta_batch_size = trial.suggest_categorical('meta_batch_size', [4, 8, 16])
            self.num_epochs = trial.suggest_int('num_epochs', 50, 200)
            self.momentum = trial.suggest_float('momentum', 0.7, 0.95)

class ProtoNetConfig(ConfigBase):
    def __init__(self, trial=None):
        super().__init__(trial)
        self.algorithm = "protonet"
        self.checkpoint_dir = os.path.join(r"D:\python\data\1", "protonet")
        
        # 默认参数
        self.embedding_dim = 64
        self.num_shots = 1
        self.test_shots = 15
        self.meta_lr = 3e-3
        self.distance_metric = "euclidean"
        
        # Optuna参数
        if trial:
            self.meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-2, log=True)
            self.embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
            self.num_shots = trial.suggest_int('num_shots', 1, 5)
            self.distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'cosine'])
            self.num_epochs = trial.suggest_int('num_epochs', 50, 150)
# ----------------------------
# 增强型数据集类（保持不变）
# ----------------------------
class EnhancedMotorMetaDataset(Dataset):
    def __init__(self, config,mode="train"):
        self.config = config
        self.mode = mode
        self.class_data = {}
        self.valid_classes = []
        
        # 加载数据
        fault_folders = []
        healthy_path = None
        self.config.mode = mode 
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
# 在EnhancedMotorMetaDataset的_split_samples方法中增加更多增强方式
def _split_samples(self, data):
    samples = []
    for i in range(0, len(data)-self.config.sample_length+1, 512):
        sample = data[i:i+self.config.sample_length]
        
        # 增加随机缩放幅度
        if random.random() > 0.7:  # 调整触发概率
            scale = np.random.uniform(0.5, 1.5)  # 扩大缩放范围
            sample *= scale
            
        # 增加通道混洗
        if random.random() > 0.5:
            np.random.shuffle(sample.T)
            
        # 增加随机掩码
        if random.random() > 0.3:
            mask_len = random.randint(10, 50)
            start = random.randint(0, self.config.sample_length - mask_len)
            sample[start:start+mask_len] = 0
            
        samples.append(sample)
    return samples
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
        train_dataset = EnhancedMotorMetaDataset(config, mode="train")
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
def train_maml(model, train_loader, config, device):
    meta_optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    best_accuracy = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_meta_loss = 0.0
        total_accuracy = 0.0
        
        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)
            
            meta_optimizer.zero_grad()
            meta_loss = 0.0
            task_accuracies = []
            
            with higher.innerloop_ctx(model, meta_optimizer, 
                                    copy_initial_weights=False,
                                    track_higher_grads=True) as (fmodel, diffopt):
                for task_idx in range(config.meta_batch_size):
                    task_support = s_d[task_idx]
                    task_s_label = s_l[task_idx]
                    task_query = q_d[task_idx]
                    task_q_label = q_l[task_idx]
                    
                    # 内循环适应
                    for _ in range(config.num_inner_steps):
                        s_output = fmodel(task_support)
                        s_loss = F.cross_entropy(s_output, task_s_label)
                        diffopt.step(s_loss)
                    
                    # 查询集评估
                    q_output = fmodel(task_query)
                    q_loss = F.cross_entropy(q_output, task_q_label)
                    meta_loss += q_loss / config.meta_batch_size
                    
                    with torch.no_grad():    
                        preds = q_output.argmax(dim=1)
                        accuracy = (preds == task_q_label).float().mean()
                        task_accuracies.append(accuracy.item())
                
                # 元优化
                meta_loss.backward()
                meta_optimizer.step()
            
            # 统计指标
            total_meta_loss += meta_loss.item()
            batch_accuracy = np.mean(task_accuracies)
            total_accuracy += batch_accuracy
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {meta_loss.item():.4f} Acc: {batch_accuracy:.2%}")

        # 保存检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': meta_optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'config': vars(config)
            }, config)
        
        # 更新最佳模型
        avg_acc = total_accuracy / len(train_loader)
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': meta_optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'config': vars(config)
            }, config, is_best=True)
# ----------------------------
#protonet
# ----------------------------
def train_protonet(model, train_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
    best_acc = 0.0
    start_epoch = 0

    # 加载检查点（保持不变）
    if config.resume_training and config.checkpoint_path:
        try:
            checkpoint = load_checkpoint(config, model, optimizer)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_accuracy', 0)
            print(f"恢复训练从epoch {start_epoch}开始，历史最佳准确率：{best_acc:.2%}")
        except Exception as e:
            print(f"加载检查点失败: {str(e)}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)  
            batch_acc = 0.0

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

                # 计算损失（保持loss为Tensor）
                loss = F.cross_entropy(-distances, query_labels)
                preds = torch.argmin(distances, dim=1) if config.distance_metric == 'euclidean' else torch.argmax(-distances, dim=1)
                acc = (preds == query_labels).float().mean()

                # 累加Tensor类型的损失，避免使用.item()
                batch_loss += loss / config.meta_batch_size
                batch_acc += acc.item() / config.meta_batch_size 

            # 反向传播
            batch_loss.backward()
            optimizer.step()

            # 累计统计（转换batch_loss为float进行记录）
            total_loss += batch_loss.item()
            total_acc += batch_acc

            # 打印进度（保持不变）
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = total_acc / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{config.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} Acc: {avg_acc:.2%}")

        # 保存检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_accuracy': best_acc,
                'config': vars(config)
            }, config)
        
        # 更新最佳模型
        epoch_acc = total_acc / len(train_loader)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_accuracy': best_acc,
                'config': vars(config)
            }, config, is_best=True)
            print(f"★ 发现新最佳模型，准确率: {best_acc:.2%}")
# ----------------------------
# Reptile训练（修正版）
# ----------------------------
def train_reptile(model, train_loader, config, device):
    meta_optimizer = optim.SGD(
    model.parameters(), 
    lr=config.meta_lr,
    momentum=config.momentum,
    weight_decay=config.weight_decay
)
    best_acc = 0.0
    start_epoch = 0

    
    # 恢复训练
    if config.resume_training and config.checkpoint_path:
        checkpoint = load_checkpoint(config, model, meta_optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_accuracy', 0)
        print(f"恢复训练从epoch {start_epoch}开始，历史最佳准确率：{best_acc:.2%}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_meta_loss = 0.0
        total_acc = 0.0
        
        for batch_idx, (s_d, s_l, q_d, q_l) in enumerate(train_loader):
            s_d, s_l = s_d.to(device), s_l.to(device)
            q_d, q_l = q_d.to(device), q_l.to(device)
            
            meta_optimizer.zero_grad()
            meta_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
            batch_loss = 0.0
            batch_acc = 0.0
            
            for task_idx in range(config.meta_batch_size):
                
                # 使用正确的优化器实例
                with higher.innerloop_ctx(
                    model, 
                    optim.SGD,
                    override={'lr':[config.inner_lr]},
                    copy_initial_weights=True,
                    track_higher_grads=False
                ) as (fmodel, diffopt):
                    # 内循环适应
                    support = s_d[task_idx]
                    labels = s_l[task_idx]
                    
                    for _ in range(config.num_inner_steps):
                        outputs, _ = fmodel(support)
                        loss = F.cross_entropy(outputs, labels)
                        diffopt.step(loss)  # 使用diffopt更新参数
                
                # 计算参数差异
                with torch.no_grad():
                    for (n, p), init_p in zip(fmodel.named_parameters(), model.parameters()):
                        meta_grads[n] += (p - init_p) / config.meta_batch_size
                
                # 查询集评估
                query = q_d[task_idx]
                q_labels = q_l[task_idx]
                outputs, _ = fmodel(query)
                loss = F.cross_entropy(outputs, q_labels)
                acc = (outputs.argmax(dim=1) == q_labels).float().mean()
                
                batch_loss += loss.item() / config.meta_batch_size
                batch_acc += acc.item() / config.meta_batch_size
            
            # 应用元梯度
            for n, p in model.named_parameters():
                if p.grad is None:
                    p.grad=torch.zeros_like(p)
                p.grad.add_(-meta_grads[n])
            meta_optimizer.step()
            
            # 累计指标
            total_meta_loss += batch_loss
            total_acc += batch_acc
            
            # 每10个batch打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_meta_loss / (batch_idx + 1)
                avg_acc = total_acc / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{config.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} Acc: {avg_acc:.2%}")

        # 计算epoch指标
        epoch_loss = total_meta_loss / len(train_loader)
        epoch_acc = total_acc / len(train_loader)
        print(f"\n== Epoch [{epoch+1}/{config.num_epochs}] 平均损失: {epoch_loss:.4f} 平均准确率: {epoch_acc:.2%} ==")
        
        # 保存检查点
        if config.save_model and (epoch + 1) % config.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': meta_optimizer.state_dict(),
                'best_accuracy': best_acc,
                'config': vars(config)
            }, config)
            print(f"检查点已保存于epoch {epoch+1}")
        
        # 更新最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': meta_optimizer.state_dict(),
                'best_accuracy': best_acc,
                'config': vars(config)
            }, config, is_best=True)
            print(f"★ 发现新最佳模型，准确率: {best_acc:.2%}")
# ----------------------------
# 模型工具函数（通用）
# ----------------------------
def save_checkpoint(state, config, is_best=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    filename = f"{config.algorithm}_"
    if is_best:
        filename += "best.pth"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename += f"epoch{state['epoch']}_{timestamp}.pth"
    
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"✓ 检查点已保存到 {filepath}")

def load_checkpoint(config, model, optimizer=None):
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    print(f"✓ 加载成功 [Epoch {checkpoint['epoch']}]")
    print(f"   最佳准确率: {checkpoint.get('best_accuracy', 0):.2%}")
    return checkpoint
# ----------------------------
# 一键训练入口
# ----------------------------
def main():
    algorithms = [ "reptile", "protonet"]
    
    for algo in algorithms:
        print(f"\n=== 开始 {algo.upper()} 自动调参 ===")
        
        # 自动调参
        trainer = AutoMetaTrainer(algo)
        best_params = trainer.optimize(n_trials=15)
        
        # 用最佳参数训练最终模型
        if algo == "maml":
            config = MAMLConfig()
            model = MetaConvModel(config)
        elif algo == "reptile":
            config = ReptileConfig()
            model = ReptileModel(config)
        else:
            config = ProtoNetConfig()
            model = ProtoNetModel(config)
        
        # 应用最佳参数
        for k, v in best_params.items():
            if hasattr(config, k):
                setattr(config, k, v)
        
        # 数据加载
        train_dataset = EnhancedMotorMetaDataset(config)
        train_loader = DataLoader(train_dataset,
                                batch_size=config.meta_batch_size,
                                shuffle=True,
                                num_workers=config.num_workers,
                                pin_memory=config.pin_memory)
        
        # 最终训练
        if algo == "maml":
            best_acc = train_maml(model, train_loader, config, config.device)
        elif algo == "reptile":
            best_acc = train_reptile(model, train_loader, config, config.device)
        else:
            best_acc = train_protonet(model, train_loader, config, config.device)
        
        print(f"{algo.upper()} 最终测试准确率: {best_acc:.2%}")

if __name__ == "__main__":
    main()