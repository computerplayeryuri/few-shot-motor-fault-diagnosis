# 小样本电机故障分类

本科毕业设计中的电机故障分类研究，尝试在样本较少时，根据振动信号识别不同故障类别。项目包含数据预处理、特征提取、元学习模型训练与测试，以及分类结果的可视化。

## 研究内容

- 使用 CWRU 轴承振动数据及三轴电机轴承振动数据开展实验。
- 以 MAML、Reptile 和 ProtoNet 为对象，尝试构建小样本分类任务。
- 使用 PyTorch 编写模型训练与测试程序，通过 Optuna 搜索部分训练参数，并生成分类报告和混淆矩阵。

## 文件

| 文件 | 内容 |
| --- | --- |
| `code/cwru_train.py` | CWRU 数据训练与参数搜索程序 |
| `code/cwru_test.py` | CWRU 测试程序 |
| `code/three_axis_train.py` | 三轴数据训练程序 |
| `code/three_axis_test.py` | 三轴数据测试程序 |

## 运行环境

原程序使用 Python、PyTorch、NumPy、SciPy、h5py、scikit-learn、Matplotlib、seaborn、Optuna 和 higher。脚本中的数据和模型路径对应原实验环境，运行前需要根据本地目录修改配置，并准备相应数据与模型权重。

不同批次的训练和测试记录还需要与具体数据划分和模型版本对应，因此本页暂不汇总跨批次的准确率。实验数据集与训练权重未包含在仓库中。
