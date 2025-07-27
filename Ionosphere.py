import numpy as np
from itertools import permutations
from scipy.stats import norm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import itertools
import math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score  # 新增：计算准确率的函数
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
import pandas as pd

class RPSGenerator:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.means = np.zeros((n_classes, n_attributes))
        self.stds = np.zeros((n_classes, n_attributes))

    def fit(self, X_train, y_train):
        """训练高斯判别模型"""
        for i in range(self.n_classes):
            class_data = X_train[y_train == i]
            self.means[i] = np.mean(class_data, axis=0)
            self.stds[i] = np.std(class_data, axis=0, ddof=1)

    def gaussian_pdf(self, x, mean, std):
        """高斯概率密度函数"""
        return norm.pdf(x, loc=mean, scale=std)

    def calculate_mv(self, x_test):
        """计算隶属度向量"""
        mv = np.zeros((self.n_attributes, self.n_classes))
        for j in range(self.n_attributes):
            for i in range(self.n_classes):
                mv[j, i] = self.gaussian_pdf(x_test[j], self.means[i, j], self.stds[i, j])
        return mv

    def calculate_nmv_onmv(self, mv):
        """计算归一化隶属度向量"""
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
                for perm in permutations(range(q)):
                    numerator = 1.0
                    for u in range(q):
                        denominator = np.sum(support[j, perm[u:]])
                        numerator *= (support[j, perm[u]] / denominator)
                    attr_weights[perm] = numerator
            weights.append(attr_weights)
        return weights

    def generate_pmf(self, onmv, weights):
        """生成概率质量函数"""
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
        """替换PMF键值为原始类别索引"""
        return [
            {tuple(onmv_indices[j][i] for i in key): value
             for key, value in prob_dist.items()}
            for j, prob_dist in enumerate(pmf)
        ]

    @staticmethod
    def generate_subsets(key):
        """生成所有非空子集排列"""
        subsets = []
        key_length = len(key)
        for subset_length in range(1, key_length + 1):
            if subset_length < key_length:
                for subset in permutations(key, subset_length):
                    subsets.append(subset)
            else:
                subsets.append(key)
        return subsets

    def redistribute_modified_pmf(self, modified_pmf):
        """重新分配PMF值"""
        redistributed = []
        for attr_pmf in modified_pmf:
            new_pmf = defaultdict(float)
            for key, value in attr_pmf.items():
                subsets = self.generate_subsets(key)
                share = value / len(subsets)
                for subset in subsets:
                    new_pmf[subset] += share
            redistributed.append(dict(new_pmf))
        return redistributed

    def generate_rps(self, X_test):
        """生成随机置换结构"""
        results = []
        for x in X_test:
            mv = self.calculate_mv(x)
            nmv, onmv, onmv_indices = self.calculate_nmv_onmv(mv)
            support = self.calculate_support(x, onmv_indices)
            weights = self.calculate_weights(support)
            pmf = self.generate_pmf(onmv, weights)
            modified_pmf = self.replace_pmf_values(onmv_indices, pmf)
            final_pmf = self.redistribute_modified_pmf(modified_pmf)
            results.append(final_pmf)
        return results

class LeaveOneOutPMFGenerator:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes

    def generate_loo_pmf(self, X_all, y_all):
        """留一法生成PMF"""
        n_samples = X_all.shape[0]
        all_pmfs = []
        for i in range(n_samples):
            X_train = np.delete(X_all, i, axis=0)
            y_train = np.delete(y_all, i)
            current_generator = RPSGenerator(self.n_classes, self.n_attributes)
            current_generator.fit(X_train, y_train)
            test_sample = X_all[i].reshape(1, -1)
            sample_pmf = current_generator.generate_rps(test_sample)
            converted_pmf = self._convert_pmf_format(sample_pmf[0])
            all_pmfs.append(converted_pmf)
        return all_pmfs

    def _convert_pmf_format(self, pmf):
        """统一PMF键格式"""
        return [
            {tuple(int(i) for i in k): v for k, v in attr_dict.items()}
            for attr_dict in pmf
        ]

class PMFAnalyzer:
    def __init__(self, pmf, N1, n_classes):
        self.pmf = pmf
        self.N1 = N1
        self.n_classes = n_classes
        self.F_ps = []
        elements = list(range(n_classes))
        max_r = min(3, n_classes)
        for r in range(1, max_r + 1):
            self.F_ps.extend(list(itertools.permutations(elements, r)))

    def compute_tau_k(self):
        """计算最优τ_k"""
        tau_k_list = []
        for k in range(len(self.pmf)):
            k_samples = self.pmf[k]
            efrps_d_values = []
            for l1 in range(self.N1):
                total_distance = 0.0
                for l2 in range(self.N1):
                    distance = 0.0
                    for F_p in self.F_ps:
                        p_l1 = k_samples[l1].get(F_p, 0.0)
                        p_l2 = k_samples[l2].get(F_p, 0.0)
                        term_l1 = p_l1 * math.log2(p_l1) if p_l1 != 0 else 0
                        term_l2 = p_l2 * math.log2(p_l2) if p_l2 != 0 else 0
                        distance += abs(term_l1 - term_l2)
                    total_distance += distance
                efrps_d_values.append(total_distance)
            tau_k = efrps_d_values.index(min(efrps_d_values))
            tau_k_list.append(tau_k)
        return tau_k_list

class PMFProcessor:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes

    def process_all_classes(self, all_pmfs, y_all):
        """处理所有类别的PMF"""
        class_pmfs = defaultdict(list)
        for pmf, label in zip(all_pmfs, y_all):
            class_pmfs[label].append(pmf)
        optimized_results = {}
        for class_label in class_pmfs:
            class_data = class_pmfs[class_label]
            reorganized_pmf = [
                [sample[k] for sample in class_data]
                for k in range(self.n_attributes)
            ]
            analyzer = PMFAnalyzer(reorganized_pmf, N1=len(class_data), n_classes=self.n_classes)
            tau_k_list = analyzer.compute_tau_k()
            optimized_pmf = [
                reorganized_pmf[k][tau_k]
                for k, tau_k in enumerate(tau_k_list)]
            optimized_results[class_label] = {
                'pmf': optimized_pmf,
                'tau_k': tau_k_list
            }
        return optimized_results

class EFRPSClassifier:
    def __init__(self, optimized_results, test_pmfs, n_classes):
        self.n_classes = n_classes
        self.n_attributes = len(test_pmfs[0])
        self.test_pmfs = test_pmfs
        self.best_pmfs = {
            c: self._reformat_pmf(optimized_results[c]['pmf'])
            for c in range(n_classes)
        }
        elements = list(range(n_classes))
        self.F_ps = []
        max_r = min(3, n_classes)
        for r in range(1, max_r + 1):
            self.F_ps.extend(list(itertools.permutations(elements, r)))

    def _reformat_pmf(self, pmf):
        return [
            {perm: prob for perm, prob in attr_pmf.items()}
            for attr_pmf in pmf
        ]

    def _compute_entropy_diff(self, pmf1, pmf2):
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
        predictions = []
        for test_pmf in self.test_pmfs:
            formatted_test = self._reformat_pmf(test_pmf)
            min_score = float('inf')
            best_class = -1
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
    # 加载Ionosphere数据集
    ionosphere = fetch_openml('ionosphere', version=1)
    X = ionosphere.data.iloc[:, 2:].values
    y = LabelEncoder().fit_transform(ionosphere.target)

    # 初始化交叉验证
    rkf = RepeatedKFold(n_splits=5, n_repeats=100, random_state=42)
    total_iterations = 5 * 100  # 总迭代次数 = 5折 × 100次重复
    accuracies = []
    reports = []
    confusion_matrices = []

    # 新增变量：用于记录每轮（5折）的召回率 (位置1)
    all_mean_recalls = []
    current_run_recalls = []

    # 添加带进度条的循环
    with tqdm(total=total_iterations, desc="Cross-Validation Progress", unit="iter") as pbar:
        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 训练模型
            n_classes = 2
            n_attributes = X_train.shape[1]
            rps_gen = RPSGenerator(n_classes, n_attributes)
            rps_gen.fit(X_train, y_train)

            # 生成测试集PMF
            test_pmfs = rps_gen.generate_rps(X_test)

            # 留一法PMF生成
            loo_generator = LeaveOneOutPMFGenerator(n_classes, n_attributes)
            all_pmfs = loo_generator.generate_loo_pmf(X_train, y_train)

            # PMF处理
            processor = PMFProcessor(n_classes, n_attributes)
            optimized_results = processor.process_all_classes(all_pmfs, y_train)

            # 分类预测
            efrps_classifier = EFRPSClassifier(optimized_results, test_pmfs, n_classes)
            y_pred = efrps_classifier.classify()

            # 收集结果
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            current_report = classification_report(y_test, y_pred, output_dict=True)  # 修改为当前report
            reports.append(current_report)

            # 新增：记录当前fold的召回率 (位置2)
            current_recall = current_report['weighted avg']['recall']
            current_run_recalls.append(current_recall)

            # 新增：每完成5折计算一次平均 (位置3)
            if len(current_run_recalls) == 5:
                mean_recall = np.mean(current_run_recalls)
                all_mean_recalls.append(mean_recall)
                current_run_recalls = []  # 重置临时存储

            confusion_matrices.append(confusion_matrix(y_test, y_pred))

            # 更新进度条（带实时准确率）
            pbar.set_postfix({
                "Last_Acc": f"{acc:.2%}",
                "Avg_Acc": f"{np.mean(accuracies):.2%}±{np.std(accuracies):.2%}"
            })
            pbar.update(1)

    # 输出统计结果（保持原有输出部分不变）
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    # ...（后续分类报告和混淆矩阵汇总代码保持不变）

    # 汇总分类报告
    avg_report = {
        '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    }

    for report in reports:
        for key in avg_report:
            for metric in ['precision', 'recall', 'f1-score']:
                avg_report[key][metric].append(report[key][metric])

    print("\nAverage Classification Report:")
    for key in avg_report:
        print(f"{key}:")
        for metric in ['precision', 'recall', 'f1-score']:
            mean_val = np.mean(avg_report[key][metric])
            std_val = np.std(avg_report[key][metric])
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    # 汇总混淆矩阵
    total_cm = np.sum(confusion_matrices, axis=0)
    print("\nTotal Confusion Matrix:")
    print(total_cm)

    # 新增：保存到Excel (位置4)
    df = pd.DataFrame({
        'Run Number': range(1, len(all_mean_recalls) + 1),
        'Average Recall': all_mean_recalls
    })
    df.to_excel('average_recall_per_run.xlsx', index=False)
    print("\n每轮平均召回率已保存至：average_recall_per_run.xlsx")