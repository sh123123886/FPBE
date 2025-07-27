import numpy as np
from itertools import permutations
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict
import itertools
import math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score


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
                        denominator = np.sum(support[j, perm[u:]])
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
        """生成键的所有非空子集排列"""
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
            results.append(modified_pmf)
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
    def __init__(self, pmf, N1, n_classes):
        """
        初始化分析器

        :param pmf: 概率质量函数列表，结构为每个属性k对应一个样本列表，每个样本是一个排列事件的概率字典
        :param N1: 第一类样本的数量
        :param n_classes: 总类别数
        """
        self.pmf = pmf
        self.N1 = N1
        self.n_classes = n_classes

        # 生成所有可能的排列事件（按照顺序：单元素、双元素、...）
        self.F_ps = []
        elements = list(range(n_classes))
        for r in range(1, n_classes + 1):
            self.F_ps.extend(list(itertools.permutations(elements, r)))

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
                        term_l1 = p_l1 if p_l1 != 0 else 0
                        term_l2 = p_l2 if p_l2 != 0 else 0

                        distance += abs(term_l1 - term_l2)
                    total_distance += distance
                efrps_d_values.append(total_distance)

            # 找到最小EFRPS_d对应的索引τ_k
            tau_k = efrps_d_values.index(min(efrps_d_values))
            tau_k_list.append(tau_k)

        return tau_k_list

class PMFProcessor:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes

    def process_all_classes(self, all_pmfs, y_all):
        """处理所有类别的PMF，返回每个类别的最优PMF字典"""
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

            # 计算τ_k (修复这里)
            analyzer = PMFAnalyzer(
                reorganized_pmf,
                N1=len(class_data),
                n_classes=self.n_classes  # 添加缺失的参数
            )
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

                term1 = - p1 if p1 != 0 else 0
                term2 = - p2 if p2 != 0 else 0

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


# 示例用法
if __name__ == "__main__":
    # 加载Sonar数据集
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    data = pd.read_csv(url, header=None)

    # 数据预处理
    X = data.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(data.iloc[:, -1])  # 将R/M转换为0/1

    # 设置参数
    n_classes = 2
    n_attributes = X.shape[1]


    total_repeats = 100
    n_splits = 5
    total_iterations = total_repeats * n_splits
    accuracies = []
    # 新增：初始化保存每轮平均召回率的列表
    repeat_recalls = []  # 修改点1：添加召回率存储列表

    with tqdm(total=total_iterations, desc="Cross-Validation Progress") as pbar:
        for repeat in range(total_repeats):
            # 新增：初始化当前重复的召回率列表
            current_repeat_recalls = []  # 修改点2：当前轮的召回率存储

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                # 数据分割
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # 模型初始化与训练
                rps_gen = RPSGenerator(n_classes, n_attributes)
                loo_generator = LeaveOneOutPMFGenerator(n_classes, n_attributes)
                rps_gen.fit(X_train, y_train)

                # 生成预测
                test_pmfs = rps_gen.generate_rps(X_test)
                all_pmfs = loo_generator.generate_loo_pmf(X_train, y_train)
                processor = PMFProcessor(n_classes, n_attributes)
                optimized_results = processor.process_all_classes(all_pmfs, y_train)
                efrps_classifier = EFRPSClassifier(optimized_results, test_pmfs, n_classes)
                y_pred = efrps_classifier.classify()

                # 计算精度
                acc = accuracy_score(y_test, y_pred)
                # 修改点3：添加召回率计算
                report = classification_report(y_test, y_pred, output_dict=True)
                recall = report['macro avg']['recall']
                current_repeat_recalls.append(recall)

                accuracies.append(acc)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "Repeat": repeat + 1,
                    "Fold": fold + 1,
                    "Avg Acc": f"{np.mean(accuracies):.4f}",  # 实时显示平均精度
                    "Last Acc": f"{acc:.4f}",  # 同时显示最后一次精度
                    "Avg Recall": f"{np.mean(current_repeat_recalls):.4f}"  # 修改点4：显示当前轮平均召回率
                })

             # 修改点5：计算并保存当前轮的平均召回率
            repeat_avg_recall = np.mean(current_repeat_recalls)
            repeat_recalls.append(repeat_avg_recall)

    # 最终统计
    print(f"\nFinal Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")

    # 修改点6：创建并保存召回率DataFrame到Excel
    recall_df = pd.DataFrame({
        'Repeat': range(1, total_repeats + 1),
        'Average Recall': repeat_recalls
    })
    recall_df.to_excel('average_recall_per_repeat1.xlsx', index=False)
    print("Average recall per repeat saved to 'average_recall_per_repeat1.xlsx'")