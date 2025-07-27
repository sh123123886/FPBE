import numpy as np
from itertools import permutations
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold  # 修改：增加交叉验证
import re
from collections import defaultdict
import itertools
import math
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm  # 新增：进度条支持
import os
import pandas as pd  # 新增：导入pandas库

# 本地数据路径配置（手动下载数据集后修改此路径）
DATA_PATH = "pima-indians-diabetes.data.csv"

# 如果检测到本地数据文件不存在，则提示手动下载
if not os.path.exists(DATA_PATH):
    print("""
    请手动执行以下步骤：
    1. 访问数据集下载页面：https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    2. 右键另存为，保存文件到当前目录下的 pima-indians-diabetes.data 文件
    3. 重新运行程序
    """)
    exit()

# 加载本地数据集
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
]

data = pd.read_csv(DATA_PATH, names=column_names)

# 数据预处理（与之前保持一致）
zero_columns = [1, 2, 3, 4, 5]  # 需要处理0值的列索引
X_all = data.iloc[:, :-1].values.astype(float)
y_all = data.iloc[:, -1].values

# 将0值替换为NaN（根据数据集文档要求）
X_all[:, zero_columns] = np.where(X_all[:, zero_columns] == 0, np.nan, X_all[:, zero_columns])

# 按类别填补缺失值（使用中位数）
for class_label in np.unique(y_all):
    class_mask = (y_all == class_label)
    class_data = X_all[class_mask]
    class_medians = np.nanmedian(class_data, axis=0)

    for col in range(X_all.shape[1]):
        nan_mask = np.isnan(X_all[:, col]) & class_mask
        X_all[nan_mask, col] = class_medians[col]


# =====================================================

class RPSGenerator:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.means = np.zeros((n_classes, n_attributes))
        self.stds = np.zeros((n_classes, n_attributes))

    def fit(self, X_train, y_train):
        """训练高斯判别模型，计算每个类别的均值和标准差"""
        for i in range(self.n_classes):
            class_data = X_train[y_train == i]
            self.means[i] = np.mean(class_data, axis=0)
            self.stds[i] = np.std(class_data, axis=0, ddof=1)

    def gaussian_pdf(self, x, mean, std):
        """高斯概率密度函数"""
        return norm.pdf(x, loc=mean, scale=std)

    def calculate_mv(self, x_test):
        """计算测试样本的隶属度向量(MV)"""
        mv = np.zeros((self.n_attributes, self.n_classes))
        for j in range(self.n_attributes):
            for i in range(self.n_classes):
                mv[j, i] = self.gaussian_pdf(x_test[j], self.means[i, j], self.stds[i, j])
        return mv

    def calculate_nmv_onmv(self, mv):
        """归一化并排序得到ONMV"""
        nmv = mv / np.sum(mv, axis=1, keepdims=True)
        onmv_indices = np.argsort(-nmv, axis=1)
        onmv = np.take_along_axis(nmv, onmv_indices, axis=1)
        return nmv, onmv, onmv_indices

    def calculate_support(self, x_test, onmv_indices):
        """计算支持度"""
        support = np.zeros_like(onmv_indices, dtype=float)
        for j in range(self.n_attributes):
            ordered_means = self.means[onmv_indices[j], j]
            support[j] = np.exp(-np.abs(x_test[j] - ordered_means))
        return support

    def calculate_weights(self, support):
        """计算权重向量"""
        weights = []
        for j in range(self.n_attributes):
            attr_weights = {}
            for q in range(1, self.n_classes + 1):
                for perm in permutations(range(q)):  ## for perm in permutations(range(self.n_classes), q):
                    numerator = 1.0
                    for u in range(q):
                        # 修改后（添加极小值防止除零）：
                        denominator = np.sum(support[j, perm[u:]]) + 1e-10
                        ## denominator = np.sum(support[j, u:q])
                        numerator *= (support[j, perm[u]] / denominator)
                    attr_weights[perm] = numerator
                    # print(u, perm[u], numerator)
            weights.append(attr_weights)
        return weights

    def generate_pmf(self, onmv, weights):
        """生成加权PMF"""
        pmf = []
        for j in range(self.n_attributes):
            attr_pmf = {}
            for perm, w in weights[j].items():
                q = len(perm)
                prob = w * onmv[j, q - 1]
                attr_pmf[perm] = prob
            pmf.append(attr_pmf)
        return pmf

    def replace_pmf_values(self, onmv_indices, pmf):
        return [
            {tuple(onmv_indices[j][i] for i in key): value
             for key, value in prob_dist.items()}
            for j, prob_dist in enumerate(pmf)
        ]

    @staticmethod
    def generate_subsets(key):
        """生成键的所有非空子集排列（包括顺序不同的情况）"""
        subsets = []
        key_length = len(key)
        for subset_length in range(1, key_length + 1):
            if subset_length < key_length:
                # 生成所有长度为 subset_length 的排列
                for subset in permutations(key, subset_length):
                    subsets.append(subset)
            else:
                # 添加父键自身
                subsets.append(key)
        return subsets

    def redistribute_modified_pmf(self, modified_pmf):
        """按照规则重新分配值，并合并相同键"""
        redistributed = []
        for attr_pmf in modified_pmf:
            new_pmf = defaultdict(float)
            for key, value in attr_pmf.items():
                # 生成所有子集排列
                subsets = self.generate_subsets(key)
                num_subsets = len(subsets)
                # 计算每个子集分得的值
                share = value / num_subsets
                # 累加到新字典（自动合并相同键）
                for subset in subsets:
                    new_pmf[subset] += share
            # 转换回普通字典
            redistributed.append(dict(new_pmf))
        return redistributed

    def generate_rps(self, X_test):
        """生成加权RPS"""
        results = []
        for x in X_test:
            mv = self.calculate_mv(x)
            nmv, onmv, onmv_indices = self.calculate_nmv_onmv(mv)
            support = self.calculate_support(x, onmv_indices)
            weights = self.calculate_weights(support)
            pmf = self.generate_pmf(onmv, weights)
            # print(pmf)
            modified_pmf = self.replace_pmf_values(onmv_indices, pmf)
            final_pmf = self.redistribute_modified_pmf(modified_pmf)
            results.append(final_pmf)
        return results


class LeaveOneOutPMFGenerator:
    def __init__(self, n_classes, n_attributes):  # 添加参数接收
        self.n_classes = n_classes
        self.n_attributes = n_attributes

    def generate_loo_pmf(self, X_all, y_all):
        """留一法生成每个样本的PMF"""
        n_samples = X_all.shape[0]
        all_pmfs = []

        for i in range(n_samples):
            # 创建临时训练集（排除当前样本）
            X_train = np.delete(X_all, i, axis=0)
            y_train = np.delete(y_all, i)

            # 初始化并训练新模型（使用类参数）
            current_generator = RPSGenerator(self.n_classes, self.n_attributes)
            current_generator.fit(X_train, y_train)

            # 生成当前测试样本的PMF
            test_sample = X_all[i].reshape(1, -1)
            sample_pmf = current_generator.generate_rps(test_sample)

            # 转换概率字典格式
            converted_pmf = self._convert_pmf_format(sample_pmf[0])
            all_pmfs.append(converted_pmf)

        return all_pmfs

    def _convert_pmf_format(self, pmf):
        """统一概率字典的键格式为元组"""
        return [
            {tuple(int(i) for i in k): v for k, v in attr_dict.items()}
            for attr_dict in pmf
        ]


class PMFAnalyzer:
    def __init__(self, pmf, N1, n_classes):  # 修改：添加n_classes参数
        self.pmf = pmf
        self.N1 = N1
        self.n_classes = n_classes  # 新增

        # 生成所有排列事件（根据实际类别数）
        self.F_ps = []
        elements = list(range(n_classes))  # 修改：动态生成元素
        for r in range(1, n_classes + 1):  # 修改：排列长度从1到n_classes
            self.F_ps.extend(list(permutations(elements, r)))

    def compute_tau_k(self):
        """
        计算每个属性k的最优τ_k

        :return: 每个属性k的τ_k索引列表
        """
        tau_k_list = []

        # 遍历每个属性k
        for k in range(len(self.pmf)):
            k_samples = self.pmf[k]  # 当前属性k的所有样本PMF

            # 计算每个样本l1的EFRPS_d值
            efrps_d_values = []
            for l1 in range(self.N1):
                # 计算与所有样本l2的总距离
                total_distance = 0.0
                for l2 in range(self.N1):
                    distance = 0.0
                    # 遍历每个排列事件F_p
                    for F_p in self.F_ps:
                        # 获取概率值，不存在则为0.0
                        p_l1 = k_samples[l1].get(F_p, 0.0)
                        p_l2 = k_samples[l2].get(F_p, 0.0)

                        # 计算熵差（处理log(0)的情况）
                        term_l1 = - p_l1 * np.log2(p_l1) if p_l1 != 0 else 0
                        term_l2 = - p_l2 * np.log2(p_l2) if p_l2 != 0 else 0

                        distance += abs(term_l1 - term_l2)
                    total_distance += distance
                efrps_d_values.append(total_distance)

            # 找到最小EFRPS_d对应的索引τ_k
            tau_k = efrps_d_values.index(min(efrps_d_values))
            tau_k_list.append(tau_k)

        return tau_k_list


class PMFProcessor:
    def __init__(self, n_classes, n_attributes):
        """
        参数说明:
        n_classes: 总类别数（如鸢尾花数据集为3）
        n_attributes: 属性数量（如鸢尾花为4）
        """
        self.n_classes = n_classes
        self.n_attributes = n_attributes

    def process_all_classes(self, all_pmfs, y_all):
        """
        处理所有类别的PMF，返回每个类别的最优PMF字典
        """
        class_pmfs = defaultdict(list)
        for pmf, label in zip(all_pmfs, y_all):
            class_pmfs[label].append(pmf)

        optimized_results = {}
        for class_label in range(self.n_classes):
            # 重组当前类别的PMF结构
            class_data = class_pmfs[class_label]
            reorganized_pmf = [
                [sample[k] for sample in class_data]
                for k in range(self.n_attributes)
            ]

            # 计算τ_k
            # 修改后（添加n_classes参数）：
            analyzer = PMFAnalyzer(reorganized_pmf, N1=len(class_data), n_classes=self.n_classes)
            tau_k_list = analyzer.compute_tau_k()

            # 生成优化后的PMF
            optimized_pmf = [
                reorganized_pmf[k][tau_k]
                for k, tau_k in enumerate(tau_k_list)
            ]

            optimized_results[class_label] = {
                'pmf': optimized_pmf,
                'tau_k': tau_k_list
            }

        return optimized_results


class EFRPSClassifier:
    def __init__(self, optimized_results, test_pmfs, n_classes=2):
        """
        初始化分类器
        :param optimized_results: PMFProcessor的处理结果
        :param test_pmfs: 测试集的PMF列表
        :param n_classes: 总类别数
        """
        self.n_classes = n_classes
        self.n_attributes = len(test_pmfs[0])
        self.test_pmfs = test_pmfs

        # 准备最佳PMF数据
        self.best_pmfs = {
            c: self._reformat_pmf(optimized_results[c]['pmf'])
            for c in range(n_classes)
        }

        # 生成标准排列顺序（动态生成）
        self.F_ps = []
        elements = list(range(n_classes))
        for r in range(1, n_classes + 1):
            self.F_ps.extend(list(permutations(elements, r)))

    def _reformat_pmf(self, pmf):
        """将PMF格式转换为属性优先的字典结构"""
        return [
            {perm: prob for perm, prob in attr_pmf.items()}
            for attr_pmf in pmf
        ]

    def _compute_entropy_diff(self, pmf1, pmf2):
        """计算两个PMF之间的熵差绝对值之和"""
        total = 0.0
        for k in range(self.n_attributes):
            for F_p in self.F_ps:
                p1 = pmf1[k].get(F_p, 0.0)
                p2 = pmf2[k].get(F_p, 0.0)

                term1 = - p1 * np.log2(p1) if p1 != 0 else 0
                term2 = - p2 * np.log2(p2) if p2 != 0 else 0

                total += abs(term1 - term2)
        return total

    def classify(self):
        """执行分类并返回预测结果"""
        predictions = []
        for test_pmf in self.test_pmfs:
            min_score = float('inf')
            best_class = -1

            # 统一测试PMF格式
            formatted_test = self._reformat_pmf(test_pmf)

            # 遍历所有候选类别
            for class_label in range(self.n_classes):
                score = self._compute_entropy_diff(
                    self.best_pmfs[class_label],
                    formatted_test
                )

                if score < min_score:
                    min_score = score
                    best_class = class_label

            predictions.append(best_class)
        return predictions


if __name__ == "__main__":
    n_classes = 2
    n_attributes = X_all.shape[1]  # 自动获取属性数量（修改后应为8）

    # 新增：初始化交叉验证
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=42)
    all_recalls = []
    current_round = []
    avg_recalls = []  # 新增：用于存储每轮平均召回率的列表

    # 修改：添加结果收集列表
    fold_results = []

    # 新增：带进度条的交叉验证循环
    with tqdm(total=500, desc="交叉验证进度") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            # 数据分割
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            # 原有模型流程
            rps_gen = RPSGenerator(n_classes, n_attributes)
            loo_generator = LeaveOneOutPMFGenerator(n_classes, n_attributes)

            rps_gen.fit(X_train, y_train)
            test_pmfs = rps_gen.generate_rps(X_test)
            all_pmfs = loo_generator.generate_loo_pmf(X_train, y_train)
            processor = PMFProcessor(n_classes, n_attributes)
            optimized_results = processor.process_all_classes(all_pmfs, y_train)

            efrps_classifier = EFRPSClassifier(optimized_results, test_pmfs, n_classes=2)
            y_pred = efrps_classifier.classify()

            # 计算评估指标
            report = classification_report(y_test, y_pred, output_dict=True)
            weighted_recall = report['weighted avg']['recall']
            current_round.append(weighted_recall)

            # 修改：直接收集所有结果
            fold_results.append(weighted_recall)

            # 新增：每完成5折打印一次结果
            if (fold_idx + 1) % 5 == 0:
                avg_recall_5fold = np.mean(fold_results[-5:])
                avg_recalls.append(avg_recall_5fold)
                print(f"\n轮次 {(fold_idx + 1) // 5}/100 平均召回率: {avg_recall_5fold:.4f}")

            pbar.update(1)

    # 修改：最终统计计算方式
    final_avg = np.mean(fold_results) * 100
    final_std = np.std(fold_results) * 100
    print(f"\nAvg_Acc={final_avg:.2f}%±{final_std:.2f}%")
