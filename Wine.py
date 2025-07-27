import numpy as np
from itertools import permutations
from scipy.stats import norm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict
import itertools
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import os
import csv
import pandas as pd

class RPSGenerator:
    def __init__(self, n_classes, n_attributes):
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.means = np.zeros((n_classes, n_attributes))
        self.stds = np.zeros((n_classes, n_attributes))

    def fit(self, X_train, y_train):
        for i in range(self.n_classes):
            class_data = X_train[y_train == i]
            self.means[i] = np.mean(class_data, axis=0)
            self.stds[i] = np.std(class_data, axis=0, ddof=1)

    def gaussian_pdf(self, x, mean, std):
        return norm.pdf(x, loc=mean, scale=std)

    def calculate_mv(self, x_test):
        mv = np.zeros((self.n_attributes, self.n_classes))
        for j in range(self.n_attributes):
            for i in range(self.n_classes):
                mv[j, i] = self.gaussian_pdf(x_test[j], self.means[i, j], self.stds[i, j])
        return mv

    def calculate_nmv_onmv(self, mv):
        nmv = mv / np.sum(mv, axis=1, keepdims=True)
        onmv_indices = np.argsort(-nmv, axis=1)
        onmv = np.take_along_axis(nmv, onmv_indices, axis=1)
        return nmv, onmv, onmv_indices

    def calculate_support(self, x_test, onmv_indices):
        support = np.zeros_like(onmv_indices, dtype=float)
        for j in range(self.n_attributes):
            ordered_means = self.means[onmv_indices[j], j]
            support[j] = np.exp(-np.abs(x_test[j] - ordered_means))
        return support

    def calculate_weights(self, support):
        weights = []
        for j in range(self.n_attributes):
            attr_weights = {}
            for q in range(1, self.n_classes + 1):
                for perm in permutations(range(self.n_classes), q):
                    numerator = 1.0
                    for u in range(q):
                        denominator = np.sum(support[j, perm[u:]])
                        numerator *= (support[j, perm[u]] / denominator)
                    attr_weights[perm] = numerator
            weights.append(attr_weights)
        return weights

    def generate_pmf(self, onmv, weights):
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
        redistributed = []
        for attr_pmf in modified_pmf:
            new_pmf = defaultdict(float)
            for key, value in attr_pmf.items():
                subsets = self.generate_subsets(key)
                num_subsets = len(subsets)
                share = value / num_subsets
                for subset in subsets:
                    new_pmf[subset] += share
            redistributed.append(dict(new_pmf))
        return redistributed

    def generate_rps(self, X_test):
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
        for r in range(1, n_classes + 1):
            self.F_ps.extend(list(itertools.permutations(elements, r)))

    def compute_tau_k(self):
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
                        term_l1 = - p_l1 * math.log2(p_l1) if p_l1 != 0 else 0
                        term_l2 = - p_l2 * math.log2(p_l2) if p_l2 != 0 else 0
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
        class_pmfs = defaultdict(list)
        for pmf, label in zip(all_pmfs, y_all):
            class_pmfs[label].append(pmf)
        optimized_results = {}
        for class_label in range(self.n_classes):
            class_data = class_pmfs[class_label]
            reorganized_pmf = [
                [sample[k] for sample in class_data]
                for k in range(self.n_attributes)
            ]
            analyzer = PMFAnalyzer(reorganized_pmf, N1=len(class_data), n_classes=self.n_classes)
            tau_k_list = analyzer.compute_tau_k()
            optimized_pmf = [
                reorganized_pmf[k][tau_k]
                for k, tau_k in enumerate(tau_k_list)
            ]
            optimized_results[class_label] = {'pmf': optimized_pmf, 'tau_k': tau_k_list}
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
        self.F_ps = []
        elements = list(range(n_classes))
        for r in range(1, n_classes + 1):
            self.F_ps.extend(list(permutations(elements, r)))

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
                score = self._compute_entropy_diff(self.best_pmfs[class_label], formatted_test)
                if score < min_score:
                    min_score = score
                    best_class = class_label
            predictions.append(best_class)
        return predictions


if __name__ == "__main__":
    wine = datasets.load_wine()
    X_all = wine.data
    y_all = wine.target

    # 数据标准化
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    n_classes = len(np.unique(y_all))
    n_attributes = X_all.shape[1]

    # 实验参数设置
    n_runs = 100  # 总实验次数
    n_folds = 5  # 交叉验证折数
    results = []  # 存储所有结果

    # 添加进度显示
    from tqdm import tqdm

    for run in tqdm(range(n_runs), desc="总体进度"):
        # 初始化当前实验的结果容器
        run_results = {
            'run': run + 1,
            'folds': [],
            'mean_recall': 0
        }

        # 创建分层交叉验证器（确保每折类别分布相同）
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=run  # 用运行次数作为随机种子
        )

        # 执行5折交叉验证
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
            # 数据划分
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            try:
                # 训练RPS生成器
                rps_gen = RPSGenerator(n_classes, n_attributes)
                rps_gen.fit(X_train, y_train)

                # 生成测试集PMF（批量处理提升效率）
                test_pmfs = rps_gen.generate_rps(X_test)

                # 生成留一法PMF（优化内存使用）
                loo_generator = LeaveOneOutPMFGenerator(n_classes, n_attributes)
                all_pmfs = loo_generator.generate_loo_pmf(X_train, y_train)

                # 处理PMF数据
                processor = PMFProcessor(n_classes, n_attributes)
                optimized_results = processor.process_all_classes(all_pmfs, y_train)

                # 执行分类
                classifier = EFRPSClassifier(optimized_results, test_pmfs, n_classes)
                y_pred = classifier.classify()

                # 计算评估指标
                report = classification_report(
                    y_test, y_pred,
                    output_dict=True,
                    zero_division=0
                )
                fold_result = {
                    'fold': fold + 1,
                    'recall': report['weighted avg']['recall'],
                    'details': report
                }

            except Exception as e:
                print(f"Run {run + 1} Fold {fold + 1} 发生错误: {str(e)}")
                fold_result = {
                    'fold': fold + 1,
                    'recall': 0,
                    'error': str(e)
                }

            run_results['folds'].append(fold_result)

        # 计算本轮平均召回率
        recalls = [f['recall'] for f in run_results['folds']]
        run_results['mean_recall'] = np.mean(recalls)
        results.append(run_results)

        # 实时输出进度
        print(f"Run {run + 1:03d} | 平均召回率: {run_results['mean_recall']:.4f} | "
              f"各折结果: {[f'{r:.4f}' for r in recalls]}")

    # 最终统计报告
    all_recalls = [r['mean_recall'] for r in results]
    print("\n最终统计报告:")
    print(f"总运行次数: {n_runs}")
    print(f"平均加权召回率: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
    print(f"最佳运行结果: {np.max(all_recalls):.4f}")
    print(f"最差运行结果: {np.min(all_recalls):.4f}")

  # 将每轮的平均召回率保存到Excel文件
    df = pd.DataFrame({
        'Run Number': [result['run'] for result in results],
        'Average Recall': [result['mean_recall'] for result in results]
    })
    df.to_excel('average_recall_results1.xlsx', index=False)
#     # 保存详细结果到文件（可选）
#     import pandas as pd
#
#     df = pd.DataFrame([{
#         'Run': r['run'],
#         'Mean_Recall': r['mean_recall'],
#         **{f'Fold_{i + 1}': f['recall'] for i, f in enumerate(r['folds'])}
#         for r in results
# ])
#     df.to_csv('cross_validation_results.csv', index=False)