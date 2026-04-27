"""
长尾噪声标签样本分析与可视化系统
支持多数据集动态适配 - 类名从数据集实际读取
支持所有真实数据集的长尾采样（不平衡分布）
"""

import os
import warnings
import logging
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from scipy.special import rel_entr
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from PIL import Image
import requests

# 配置
os.environ['TORCH_HOME'] = './torch_cache'
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

st.set_page_config(page_title="长尾噪声分析系统", layout="wide")

# 样式
st.markdown("""
<style>
.main-header { font-size: 2rem; color: #1E88E5; text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px; }
.sub-header { font-size: 1.3rem; color: #0d47a1; padding: 0.5rem; border-left: 5px solid #1E88E5; background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)


# ==================== 数据集加载类 ====================

class DatasetLoader:
    """数据集加载器 - 支持长尾采样"""

    @staticmethod
    def load_cifar10n(data_dir='./data', noise_type='worst', num_samples=None, imbalance_ratio=None, head_samples=None):
        """
        加载 CIFAR-10N 数据集
        支持普通随机采样和长尾采样
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=transform
        )

        class_names_display = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

        # 加载噪声标签文件
        npy_path = os.path.join(data_dir, 'CIFAR-10_human_ordered.npy')
        if not os.path.exists(npy_path):
            st.error(f"找不到噪声标签文件: {npy_path}")
            return None, None, None, None

        human_annotations = np.load(npy_path, allow_pickle=True).item()
        clean_labels = human_annotations['clean_label']
        label_map = {
            'aggregate': 'aggre_label',
            'random1': 'random_label1',
            'random2': 'random_label2',
            'random3': 'random_label3',
            'worst': 'worse_label'
        }
        noisy_labels_all = human_annotations[label_map[noise_type]]

        # 原始标签（用于长尾采样）
        original_targets = np.array(train_dataset.targets)
        total_samples = len(train_dataset)
        num_classes = 10

        # ========== 采样策略 ==========
        if imbalance_ratio is not None and imbalance_ratio > 0:
            # 长尾采样
            head_max = head_samples if head_samples else 1000
            class_counts = []
            for i in range(num_classes):
                ratio = imbalance_ratio ** (i / (num_classes - 1))
                count = max(1, int(head_max * ratio))
                class_counts.append(count)

            indices_per_class = []
            for c in range(num_classes):
                class_indices = np.where(original_targets == c)[0]
                if len(class_indices) == 0:
                    st.error(f"类别 {c} 在原始数据中没有样本")
                    return None, None, None, None
                if len(class_indices) < class_counts[c]:
                    st.warning(f"类别 {c} 原始样本不足（需要 {class_counts[c]}，实际 {len(class_indices)}），使用全部 {len(class_indices)} 个")
                    class_counts[c] = len(class_indices)
                indices_per_class.append(np.random.choice(class_indices, class_counts[c], replace=False))
            indices = np.concatenate(indices_per_class)
            st.info(f"长尾采样完成，总样本数: {len(indices)}，不平衡率: {imbalance_ratio:.3f}")
        else:
            # 普通随机采样
            if num_samples is None or num_samples >= total_samples:
                indices = range(total_samples)
            else:
                indices = np.random.choice(total_samples, num_samples, replace=False)
            st.info(f"随机采样: {len(indices)} 个样本")

        samples = []
        labels_noisy = []
        labels_true = []

        for idx in indices:
            img, _ = train_dataset[idx]
            samples.append({
                'id': f'cifar10n_{idx:05d}',
                'image': img,
                'class_name': class_names_display[noisy_labels_all[idx] % 10]
            })
            labels_noisy.append(noisy_labels_all[idx] % 10)
            labels_true.append(clean_labels[idx])

        actual_noise_rate = (np.array(labels_noisy) != np.array(labels_true)).mean()
        st.info(f"实际噪声率: {actual_noise_rate:.2%}")

        return samples, np.array(labels_noisy), np.array(labels_true), class_names_display

    @staticmethod
    def load_cifar100n(data_dir='./data', num_samples=None, imbalance_ratio=None, head_samples=None):
        """
        加载 CIFAR-100N 数据集
        支持普通随机采样和长尾采样
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=transform
        )

        # 原始标签（用于长尾采样）
        original_targets = np.array(train_dataset.targets)
        class_names = train_dataset.classes
        num_classes = 100

        # 加载噪声标签
        noisy_labels_path = os.path.join(data_dir, 'cifar100n.pt')
        if not os.path.exists(noisy_labels_path):
            url = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt"
            try:
                response = requests.get(url, timeout=30)
                with open(noisy_labels_path, 'wb') as f:
                    f.write(response.content)
            except:
                st.warning("无法下载噪声标签，使用原始标签")
                noisy_labels_all = original_targets.copy()
            else:
                noisy_labels_all = torch.load(noisy_labels_path)
        else:
            noisy_labels_all = torch.load(noisy_labels_path)

        # ========== 采样策略 ==========
        if imbalance_ratio is not None and imbalance_ratio > 0:
            # 长尾采样
            head_max = head_samples if head_samples else 1000
            class_counts = []
            for i in range(num_classes):
                ratio = imbalance_ratio ** (i / (num_classes - 1))
                count = max(1, int(head_max * ratio))
                class_counts.append(count)

            indices_per_class = []
            for c in range(num_classes):
                class_indices = np.where(original_targets == c)[0]
                if len(class_indices) == 0:
                    st.error(f"类别 {c} 在原始数据中没有样本")
                    return None, None, None, None
                if len(class_indices) < class_counts[c]:
                    st.warning(f"类别 {c} 原始样本不足，使用全部 {len(class_indices)} 个")
                    class_counts[c] = len(class_indices)
                indices_per_class.append(np.random.choice(class_indices, class_counts[c], replace=False))
            indices = np.concatenate(indices_per_class)
            st.info(f"长尾采样完成，总样本数: {len(indices)}，不平衡率: {imbalance_ratio:.3f}")
        else:
            # 普通随机采样
            total_samples = len(train_dataset)
            if num_samples is None or num_samples >= total_samples:
                indices = range(total_samples)
            else:
                indices = np.random.choice(total_samples, num_samples, replace=False)
            st.info(f"随机采样: {len(indices)} 个样本")

        samples = []
        labels_noisy = []
        labels_true = []

        for idx in indices:
            img, _ = train_dataset[idx]
            samples.append({
                'id': f'cifar100n_{idx:05d}',
                'image': img,
                'class_name': class_names[noisy_labels_all[idx]]
            })
            labels_noisy.append(noisy_labels_all[idx])
            labels_true.append(original_targets[idx])

        actual_noise_rate = (np.array(labels_noisy) != np.array(labels_true)).mean()
        st.info(f"实际噪声率: {actual_noise_rate:.2%}")

        return samples, np.array(labels_noisy), np.array(labels_true), class_names

    @staticmethod
    def load_animal10n(data_dir='./data', num_samples=None, imbalance_ratio=None, head_samples=None):
        """
        加载 Animal-10N 数据集
        支持普通随机采样和长尾采样
        """
        data_path = os.path.join(data_dir, 'Animal-10N')
        samples = []
        labels_noisy = []
        labels_true = []
        class_names = []

        if os.path.exists(data_path):
            subdirs = [d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d))]
            subdirs.sort()
            class_names = subdirs
            num_classes = len(class_names)

            # 预先收集每个类别的所有图像路径
            class_image_paths = [[] for _ in range(num_classes)]
            for class_idx, class_name in enumerate(subdirs):
                class_dir = os.path.join(data_path, class_name)
                img_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                class_image_paths[class_idx] = [os.path.join(class_dir, f) for f in img_files]

            # ========== 采样策略 ==========
            if imbalance_ratio is not None and imbalance_ratio > 0:
                # 长尾采样
                head_max = head_samples if head_samples else 200
                class_counts = []
                for i in range(num_classes):
                    ratio = imbalance_ratio ** (i / (num_classes - 1))
                    count = max(1, int(head_max * ratio))
                    class_counts.append(count)

                for c in range(num_classes):
                    available = len(class_image_paths[c])
                    if available < class_counts[c]:
                        st.warning(f"类别 {class_names[c]} 原始样本不足（需要 {class_counts[c]}，实际 {available}），使用全部 {available} 个")
                        class_counts[c] = available
            else:
                # 普通采样
                if num_samples is None:
                    class_counts = [len(class_image_paths[c]) for c in range(num_classes)]
                else:
                    per_class = num_samples // num_classes + 1
                    class_counts = [min(per_class, len(class_image_paths[c])) for c in range(num_classes)]
                    total = sum(class_counts)
                    if total > num_samples:
                        diff = total - num_samples
                        for c in range(num_classes - 1, -1, -1):  # 从尾部开始减少
                            if diff <= 0:
                                break
                            reduce = min(diff, class_counts[c] - 1)
                            class_counts[c] -= reduce
                            diff -= reduce

            # 采样
            for c in range(num_classes):
                paths = class_image_paths[c]
                count = class_counts[c]
                if count <= 0:
                    continue
                selected_paths = np.random.choice(paths, count, replace=False) if count < len(paths) else paths
                for img_path in selected_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = transforms.Resize((224, 224))(img)
                        img = transforms.ToTensor()(img)
                        samples.append({
                            'id': f'animal_{c}_{os.path.basename(img_path)}',
                            'image': img,
                            'class_name': class_names[c]
                        })
                        labels_noisy.append(c)
                        labels_true.append(c)
                    except:
                        continue

            total_samples = len(samples)
            if imbalance_ratio:
                st.info(f"长尾采样完成，总样本数: {total_samples}，不平衡率: {imbalance_ratio:.3f}")
            else:
                st.info(f"随机采样完成，总样本数: {total_samples}")
        else:
            st.warning(f"Animal-10N 数据不存在: {data_path}，使用模拟数据")
            class_names = ['dog', 'cat', 'horse', 'sheep', 'cow', 'elephant',
                           'bear', 'zebra', 'giraffe', 'pig']
            num_classes = len(class_names)
            if imbalance_ratio is not None:
                # 模拟长尾采样
                head_max = head_samples if head_samples else 200
                class_counts = []
                for i in range(num_classes):
                    ratio = imbalance_ratio ** (i / (num_classes - 1))
                    count = max(1, int(head_max * ratio))
                    class_counts.append(count)
            else:
                if num_samples:
                    per_class = num_samples // num_classes + 1
                    class_counts = [per_class] * num_classes
                else:
                    class_counts = [200] * num_classes

            for class_idx, count in enumerate(class_counts):
                for j in range(count):
                    samples.append({
                        'id': f'animal_{class_idx}_{j:04d}',
                        'image': torch.randn(3, 224, 224),
                        'class_name': class_names[class_idx]
                    })
                    labels_noisy.append(class_idx)
                    labels_true.append(class_idx)

        return samples, np.array(labels_noisy), np.array(labels_true), class_names


# ==================== 模拟数据生成 ====================

def generate_synthetic_data(num_samples=200, num_classes=10, imbalance=0.1, noise_rate=0.2):
    """生成模拟长尾噪声数据"""
    class_names = [f'Class_{i}' for i in range(num_classes)]
    samples = []
    true_labels = []

    max_samples = num_samples // 3
    class_counts = []
    for i in range(num_classes):
        count = int(max_samples * np.exp(-0.3 * i))
        count = max(1, count)
        class_counts.append(count)

    total = sum(class_counts)
    scale = num_samples / total
    class_counts = [max(1, int(c * scale)) for c in class_counts]
    diff = num_samples - sum(class_counts)
    if diff > 0:
        for i in range(min(diff, num_classes)):
            class_counts[i] += 1

    for class_idx, count in enumerate(class_counts):
        for j in range(count):
            samples.append({
                'id': f'sim_{class_idx}_{j:04d}',
                'image': torch.randn(3, 32, 32),
                'class_name': class_names[class_idx]
            })
            true_labels.append(class_idx)

    true_labels = np.array(true_labels)
    noisy_labels = true_labels.copy()
    num_noisy = int(num_samples * noise_rate)
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)
    for idx in noisy_indices:
        current = true_labels[idx]
        other_classes = [c for c in range(num_classes) if c != current]
        if other_classes:
            noisy_labels[idx] = np.random.choice(other_classes)

    return samples, noisy_labels, true_labels, class_names


# ==================== TABASCO 噪声检测器 ====================

class TABASCODetector:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_wjsd(self, probabilities, labels):
        n_samples = len(probabilities)
        wjsd_scores = np.zeros(n_samples)

        class_avg = []
        for c in range(self.num_classes):
            mask = labels == c
            if np.sum(mask) > 0:
                class_avg.append(np.mean(probabilities[mask], axis=0))
            else:
                class_avg.append(np.ones(self.num_classes) / self.num_classes)
        class_avg = np.array(class_avg)

        for i in range(n_samples):
            c = labels[i]
            p = np.clip(probabilities[i], 1e-10, 1)
            q = np.clip(class_avg[c], 1e-10, 1)
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)

            js_div = 0.5 * (np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m)))
            class_weight = 1.0 / np.log(np.sum(labels == c) + 2)
            wjsd_scores[i] = js_div * class_weight

        return wjsd_scores

    def compute_acd(self, features, labels, confidences):
        n_samples = len(features)
        acd_scores = np.zeros(n_samples)

        class_centroids = []
        for c in range(self.num_classes):
            mask = labels == c
            if np.sum(mask) > 0:
                high_conf_mask = mask & (confidences > 0.7)
                if np.sum(high_conf_mask) > 0:
                    centroid = np.mean(features[high_conf_mask], axis=0)
                else:
                    centroid = np.mean(features[mask], axis=0)
            else:
                centroid = np.zeros(features.shape[1])
            class_centroids.append(centroid)

        class_centroids = np.array(class_centroids)

        for i in range(n_samples):
            c = labels[i]
            if np.linalg.norm(features[i]) > 0 and np.linalg.norm(class_centroids[c]) > 0:
                dist = 1 - np.dot(features[i], class_centroids[c]) / (
                        np.linalg.norm(features[i]) * np.linalg.norm(class_centroids[c]) + 1e-8)
            else:
                dist = 1.0

            class_size = np.sum(labels == c)
            adaptive = 1.0 / (1.0 + np.log(class_size + 1))
            acd_scores[i] = dist * adaptive

        return acd_scores

    def _dimension_selection(self, wjsd_scores, acd_scores):
        """完整维度选择策略（基于GMM）"""
        wjsd_reshaped = wjsd_scores.reshape(-1, 1)
        gmm_wjsd = GaussianMixture(n_components=2, random_state=42)
        gmm_wjsd.fit(wjsd_reshaped)
        means_wjsd = gmm_wjsd.means_.flatten()
        d = (means_wjsd[0] + means_wjsd[1]) / 2

        acd_reshaped = acd_scores.reshape(-1, 1)
        gmm_acd = GaussianMixture(n_components=2, random_state=42)
        cluster_labels_acd = gmm_acd.fit_predict(acd_reshaped)

        cluster0_mask = (cluster_labels_acd == 0)
        cluster1_mask = (cluster_labels_acd == 1)
        wjsd_cluster0 = wjsd_scores[cluster0_mask]
        wjsd_cluster1 = wjsd_scores[cluster1_mask]

        mu1 = wjsd_cluster0.mean() if len(wjsd_cluster0) > 0 else np.inf
        mu2 = wjsd_cluster1.mean() if len(wjsd_cluster1) > 0 else np.inf
        sigma1 = wjsd_cluster0.std() if len(wjsd_cluster0) > 0 else np.inf
        sigma2 = wjsd_cluster1.std() if len(wjsd_cluster1) > 0 else np.inf

        eta = 0.5
        if (mu1 < d < mu2 or mu2 < d < mu1) and (sigma2 / sigma1 < eta or sigma1 / sigma2 < eta):
            return wjsd_scores, True
        if mu1 > d and mu2 > d:
            return wjsd_scores, True
        return acd_scores, False

    def detect(self, probabilities, features, labels, confidences):
        n_samples = len(labels)

        wjsd = self.compute_wjsd(probabilities, labels)
        acd = self.compute_acd(features, labels, confidences)

        wjsd = (wjsd - wjsd.min()) / (wjsd.max() - wjsd.min() + 1e-8)
        acd = (acd - acd.min()) / (acd.max() - acd.min() + 1e-8)

        sample_types = np.zeros(n_samples, dtype=int)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue

            idx = np.where(mask)[0]
            c_wjsd = wjsd[idx]
            c_acd = acd[idx]

            scores, use_wjsd = self._dimension_selection(c_wjsd, c_acd)

            scores_reshaped = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42)
            cluster_labels = gmm.fit_predict(scores_reshaped)

            mean0 = scores[cluster_labels == 0].mean() if np.sum(cluster_labels == 0) > 0 else np.inf
            mean1 = scores[cluster_labels == 1].mean() if np.sum(cluster_labels == 1) > 0 else np.inf
            clean_cluster = 0 if mean0 < mean1 else 1

            clean_mask = (cluster_labels == clean_cluster)
            clean_scores = scores[clean_mask]

            if len(clean_scores) > 0:
                q25_clean = np.percentile(clean_scores, 25)
                for i, pos in enumerate(idx):
                    if cluster_labels[i] == clean_cluster:
                        if scores[i] <= q25_clean:
                            sample_types[pos] = 0
                        else:
                            sample_types[pos] = 1
                    else:
                        sample_types[pos] = 2
            else:
                for pos in idx:
                    sample_types[pos] = 1

        return sample_types, wjsd, acd


# ==================== 模型管理 ====================

class RealModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = None

    def load_model(self, model_name='resnet18', num_classes=10):
        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=False)
        elif model_name == 'resnet34':
            self.model = torchvision.models.resnet34(pretrained=False)
        else:
            self.model = torchvision.models.resnet50(pretrained=False)

        weight_files = {
            'resnet18': 'resnet18-f37072fd.pth',
            'resnet34': 'resnet34-b627a593.pth',
            'resnet50': 'resnet50-0676ba61.pth'
        }
        cache_dir = './torch_cache/hub/checkpoints'
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, weight_files[model_name])

        if os.path.exists(local_path):
            try:
                state_dict = torch.load(local_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                st.success(f"✓ {model_name} 权重加载成功")
            except Exception as e:
                st.warning(f"加载权重失败: {e}")
        else:
            st.warning(f"本地模型文件不存在: {local_path}")

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self.model

    def extract_features(self, images):
        features = []
        with torch.no_grad():
            for sample in images:
                img = sample['image']
                if img.shape[1] <= 32:
                    img = transforms.Resize(224)(img)
                img = img.unsqueeze(0).to(self.device)
                x = self.model.conv1(img)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                feat = x.flatten().cpu().numpy()
                features.append(feat)
        return np.array(features)

    def predict(self, images):
        probs_list, preds_list, confs_list = [], [], []
        with torch.no_grad():
            for sample in images:
                img = sample['image']
                if img.shape[1] <= 32:
                    img = transforms.Resize(224)(img)
                img = img.unsqueeze(0).to(self.device)
                output = self.model(img)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
                conf = probs[pred]
                probs_list.append(probs)
                preds_list.append(pred)
                confs_list.append(conf)
        return np.array(probs_list), np.array(preds_list), np.array(confs_list)


class MockModelManager:
    def extract_features(self, images):
        return np.random.randn(len(images), 512)
    def predict(self, images):
        n = len(images)
        probs = np.random.randn(n, 10)
        probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
        return probs, np.argmax(probs, axis=1), np.max(probs, axis=1)


# ==================== 可视化函数 ====================

def plot_class_distribution(labels, class_names, max_display=20):
    counter = Counter(labels)
    if len(class_names) > max_display:
        display_names = class_names[:max_display]
        counts = [counter.get(i, 0) for i in range(max_display)]
    else:
        display_names = class_names
        counts = [counter.get(i, 0) for i in range(len(class_names))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=display_names, y=counts, marker_color='#1E88E5', text=counts, textposition='auto'))
    fig.update_layout(title="类别分布", xaxis_tickangle=-45, height=500)
    return fig

def plot_sample_type_distribution(sample_types, labels, class_names, max_display=20):
    if sample_types is None or labels is None:
        fig = go.Figure()
        fig.update_layout(title="各类别样本类型分布（暂无数据）", height=500)
        return fig
    if len(sample_types) != len(labels):
        min_len = min(len(sample_types), len(labels))
        sample_types = sample_types[:min_len]
        labels = labels[:min_len]
    if len(sample_types) == 0:
        fig = go.Figure()
        fig.update_layout(title="各类别样本类型分布（无数据）", height=500)
        return fig
    type_names = ['高可信', '低可信', '疑似噪声']
    colors = ['#4caf50', '#ff9800', '#f44336']
    if len(class_names) > max_display:
        display_names = class_names[:max_display]
    else:
        display_names = class_names
    data = []
    for i in range(3):
        counts = [np.sum((labels == c) & (sample_types == i)) for c in range(len(display_names))]
        data.append(go.Bar(name=type_names[i], x=display_names, y=counts, marker_color=colors[i]))
    fig = go.Figure(data=data)
    fig.update_layout(title="各类别样本类型分布", barmode='stack', xaxis_tickangle=-45, height=500)
    return fig

def plot_confidence_distribution(confidences, sample_types):
    fig = go.Figure()
    type_names = ['高可信', '低可信', '疑似噪声']
    colors = ['#4caf50', '#ff9800', '#f44336']
    for i in range(3):
        mask = sample_types == i
        if mask.sum() > 0:
            fig.add_trace(go.Histogram(x=confidences[mask], name=type_names[i],
                                       marker_color=colors[i], opacity=0.7, nbinsx=20))
    fig.update_layout(title="置信度分布", barmode='overlay', height=400)
    return fig

def plot_confusion_matrix(true_labels, pred_labels, class_names, max_display=15):
    if len(class_names) > max_display:
        display_names = class_names[:max_display]
        true_display = np.clip(true_labels, 0, max_display - 1)
        pred_display = np.clip(pred_labels, 0, max_display - 1)
    else:
        display_names = class_names
        true_display = true_labels
        pred_display = pred_labels
    cm = confusion_matrix(true_display, pred_display)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig = px.imshow(cm_norm, x=display_names, y=display_names,
                    text_auto='.2f', color_continuous_scale='Blues', aspect="auto")
    fig.update_layout(title="混淆矩阵", height=500)
    return fig


# ==================== 主程序 ====================

def main():
    st.markdown('<h1 class="main-header">📊 长尾噪声标签分析系统</h1>', unsafe_allow_html=True)

    # 状态初始化
    for key in ['samples', 'labels', 'true_labels', 'class_names', 'predictions',
                'confidences', 'sample_types', 'modified_labels', 'features']:
        if key not in st.session_state:
            st.session_state[key] = None

    # 侧边栏
    with st.sidebar:
        st.markdown("## 🔧 配置")

        dataset_type = st.selectbox("数据集类型", [
            "CIFAR-10N (真实噪声)",
            "CIFAR-100N (真实噪声)",
            "Animal-10N (真实噪声)",
            "CIFAR-10-LT (模拟)"
        ])

        # ==================== CIFAR-10N ====================
        if dataset_type == "CIFAR-10N (真实噪声)":
            noise_type = st.selectbox("噪声类型", ["worst", "aggregate", "random1", "random2", "random3"])

            enable_longtail = st.checkbox("构造长尾分布（按不平衡率采样）", value=False)
            if enable_longtail:
                imbalance_ratio = st.slider("不平衡率 (尾部/头部样本比例)", 0.01, 0.5, 0.1, 0.01)
                head_samples = st.number_input("头部类别采样数量", min_value=10, max_value=5000, value=1000)
                num_samples = None
            else:
                imbalance_ratio = None
                head_samples = None
                sample_mode = st.radio("选择模式", ["指定数量", "全部数据"], horizontal=True)
                if sample_mode == "指定数量":
                    num_samples = st.number_input("样本数量", min_value=10, max_value=50000, value=200, step=50)
                else:
                    num_samples = None

            if st.button("加载数据集", type="primary"):
                with st.spinner("加载 CIFAR-10N..."):
                    result = DatasetLoader.load_cifar10n(
                        num_samples=num_samples,
                        noise_type=noise_type,
                        imbalance_ratio=imbalance_ratio if enable_longtail else None,
                        head_samples=head_samples if enable_longtail else None
                    )
                    if result[0] is not None:
                        samples, labels_noisy, labels_true, class_names = result
                        st.session_state.clear()
                        st.session_state.samples = samples
                        st.session_state.labels = labels_noisy
                        st.session_state.true_labels = labels_true
                        st.session_state.class_names = class_names
                        st.session_state.modified_labels = labels_noisy.copy()
                        st.session_state.predictions = None
                        st.session_state.confidences = None
                        st.session_state.sample_types = None
                        st.session_state.features = None
                        st.success(f"✅ 加载完成: {len(samples)} 样本, {len(class_names)} 类别")

        # ==================== CIFAR-100N ====================
        elif dataset_type == "CIFAR-100N (真实噪声)":
            enable_longtail = st.checkbox("构造长尾分布（按不平衡率采样）", value=False)
            if enable_longtail:
                imbalance_ratio = st.slider("不平衡率 (尾部/头部样本比例)", 0.01, 0.5, 0.1, 0.01)
                head_samples = st.number_input("头部类别采样数量", min_value=10, max_value=3000, value=500)
                num_samples = None
            else:
                imbalance_ratio = None
                head_samples = None
                sample_mode = st.radio("选择模式", ["指定数量", "全部数据"], horizontal=True)
                if sample_mode == "指定数量":
                    num_samples = st.number_input("样本数量", min_value=10, max_value=50000, value=200, step=50)
                else:
                    num_samples = None

            if st.button("加载数据集", type="primary"):
                with st.spinner("加载 CIFAR-100N..."):
                    result = DatasetLoader.load_cifar100n(
                        num_samples=num_samples,
                        imbalance_ratio=imbalance_ratio if enable_longtail else None,
                        head_samples=head_samples if enable_longtail else None
                    )
                    if result[0] is not None:
                        samples, labels_noisy, labels_true, class_names = result
                        st.session_state.clear()
                        st.session_state.samples = samples
                        st.session_state.labels = labels_noisy
                        st.session_state.true_labels = labels_true
                        st.session_state.class_names = class_names
                        st.session_state.modified_labels = labels_noisy.copy()
                        st.session_state.predictions = None
                        st.session_state.confidences = None
                        st.session_state.sample_types = None
                        st.session_state.features = None
                        st.success(f"✅ 加载完成: {len(samples)} 样本, {len(class_names)} 类别")

        # ==================== Animal-10N ====================
        elif dataset_type == "Animal-10N (真实噪声)":
            enable_longtail = st.checkbox("构造长尾分布（按不平衡率采样）", value=False)
            if enable_longtail:
                imbalance_ratio = st.slider("不平衡率 (尾部/头部样本比例)", 0.01, 0.5, 0.1, 0.01)
                head_samples = st.number_input("头部类别采样数量", min_value=10, max_value=1000, value=200)
                num_samples = None
            else:
                imbalance_ratio = None
                head_samples = None
                sample_mode = st.radio("选择模式", ["指定数量", "全部数据"], horizontal=True)
                if sample_mode == "指定数量":
                    num_samples = st.number_input("样本数量", min_value=10, max_value=50000, value=200, step=50)
                else:
                    num_samples = None

            if st.button("加载数据集", type="primary"):
                with st.spinner("加载 Animal-10N..."):
                    result = DatasetLoader.load_animal10n(
                        num_samples=num_samples,
                        imbalance_ratio=imbalance_ratio if enable_longtail else None,
                        head_samples=head_samples if enable_longtail else None
                    )
                    if result[0] is not None:
                        samples, labels_noisy, labels_true, class_names = result
                        st.session_state.clear()
                        st.session_state.samples = samples
                        st.session_state.labels = labels_noisy
                        st.session_state.true_labels = labels_true
                        st.session_state.class_names = class_names
                        st.session_state.modified_labels = labels_noisy.copy()
                        st.session_state.predictions = None
                        st.session_state.confidences = None
                        st.session_state.sample_types = None
                        st.session_state.features = None
                        st.success(f"✅ 加载完成: {len(samples)} 样本, {len(class_names)} 类别")

        # ==================== 模拟数据 ====================
        else:
            num_classes = st.number_input("类别数量", 2, 50, 10)
            imbalance = st.slider("长尾不平衡率", 0.05, 0.5, 0.1, 0.05)
            noise_rate = st.slider("噪声率", 0.0, 0.5, 0.2, 0.05)
            sample_mode = st.radio("选择模式", ["指定数量", "全部数据"], horizontal=True)
            if sample_mode == "指定数量":
                num_samples = st.number_input("样本数量", min_value=10, max_value=50000, value=200, step=50)
            else:
                num_samples = 200
            if st.button("生成数据集", type="primary"):
                with st.spinner("生成模拟数据..."):
                    samples, labels_noisy, labels_true, class_names = generate_synthetic_data(
                        num_samples, num_classes, imbalance, noise_rate
                    )
                    st.session_state.samples = samples
                    st.session_state.labels = labels_noisy
                    st.session_state.true_labels = labels_true
                    st.session_state.class_names = class_names
                    st.session_state.modified_labels = labels_noisy.copy()
                    st.session_state.predictions = None
                    st.session_state.confidences = None
                    st.session_state.sample_types = None
                    st.success(f"✅ 生成完成: {len(samples)} 样本, {len(class_names)} 类别")

        st.markdown("---")

        model_name = st.selectbox("模型", ["resnet18", "resnet34", "resnet50"])
        use_real = st.radio("模型模式", ["真实模型", "模拟模式"], index=0)

        if st.button("运行 TABASCO 检测", type="primary"):
            if st.session_state.samples is None:
                st.warning("请先加载数据集")
            else:
                with st.spinner("运行检测..."):
                    num_c = len(st.session_state.class_names)
                    if use_real == "真实模型":
                        mgr = RealModelManager()
                        mgr.load_model(model_name, num_c)
                    else:
                        mgr = MockModelManager()

                    features = mgr.extract_features(st.session_state.samples)
                    probs, preds, confs = mgr.predict(st.session_state.samples)

                    detector = TABASCODetector(num_c)
                    types, wjsd, acd = detector.detect(probs, features, st.session_state.labels, confs)

                    st.session_state.predictions = preds
                    st.session_state.confidences = confs
                    st.session_state.sample_types = types

                    st.success(f"✅ 检测完成: 高可信:{sum(types==0)} 低可信:{sum(types==1)} 疑似噪声:{sum(types==2)}")

    # 主内容区（省略详细代码以节省篇幅，实际使用时需保留完整内容）
    if st.session_state.samples is None:
        st.info("👈 请选择数据集并点击加载")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("样本数", len(st.session_state.samples))
    with col2: st.metric("类别数", len(st.session_state.class_names))
    with col3:
        if st.session_state.sample_types is not None:
            st.metric("疑似噪声", sum(st.session_state.sample_types == 2))
        else:
            st.metric("疑似噪声", "待检测")
    with col4:
        if st.session_state.confidences is not None:
            st.metric("平均置信度", f"{st.session_state.confidences.mean():.1%}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据分布", "🔍 噪声检测", "📈 模型性能", "✏️ 标签修正"])

    with tab1:
        st.plotly_chart(plot_class_distribution(st.session_state.labels, st.session_state.class_names), use_container_width=True)

    with tab2:
        if st.session_state.sample_types is not None:
            st.plotly_chart(plot_sample_type_distribution(st.session_state.sample_types, st.session_state.labels, st.session_state.class_names), use_container_width=True)
            st.plotly_chart(plot_confidence_distribution(st.session_state.confidences, st.session_state.sample_types), use_container_width=True)
        else:
            st.info("请先运行检测")

    with tab3:
        if st.session_state.predictions is not None:
            if st.session_state.true_labels is not None:
                acc = (st.session_state.predictions == st.session_state.true_labels).mean()
            else:
                acc = (st.session_state.predictions == st.session_state.labels).mean()
            st.metric("整体准确率", f"{acc:.2%}")
            if st.session_state.true_labels is not None:
                plot_labels = st.session_state.true_labels
            else:
                plot_labels = st.session_state.labels
            st.plotly_chart(plot_confusion_matrix(plot_labels, st.session_state.predictions, st.session_state.class_names), use_container_width=True)
        else:
            st.info("请先运行检测")

    with tab4:
        if st.session_state.sample_types is not None:
            filter_type = st.selectbox("筛选", ["全部", "高可信", "低可信", "疑似噪声"])
            type_map = {"全部": None, "高可信": 0, "低可信": 1, "疑似噪声": 2}
            if filter_type != "全部":
                indices = np.where(st.session_state.sample_types == type_map[filter_type])[0]
            else:
                indices = np.arange(len(st.session_state.samples))

            per_page = st.selectbox("每页", [5, 10, 20], 0)
            total_pages = max(1, (len(indices) + per_page - 1) // per_page)
            page = st.number_input("页码", 1, total_pages, 1)
            start, end = (page-1)*per_page, min(page*per_page, len(indices))

            for idx in indices[start:end]:
                type_name = ["高可信", "低可信", "疑似噪声"][st.session_state.sample_types[idx]]
                color = {"高可信":"green", "低可信":"orange", "疑似噪声":"red"}[type_name]
                st.markdown(f'<div style="border-left: 5px solid {color}; padding: 10px; margin: 5px 0;">', unsafe_allow_html=True)
                cols = st.columns([2,2,2,2])
                with cols[0]: st.write(f"**ID:** {st.session_state.samples[idx]['id'][:20]}...")
                with cols[1]:
                    if st.session_state.true_labels is not None:
                        true_name = st.session_state.class_names[st.session_state.true_labels[idx]]
                        st.write(f"**真实标签:** {true_name}")
                    st.write(f"**当前标签:** {st.session_state.class_names[st.session_state.modified_labels[idx]]}")
                with cols[2]: st.write(f"**置信度:** {st.session_state.confidences[idx]:.2%}")
                with cols[3]:
                    new = st.selectbox("修正为", range(len(st.session_state.class_names)),
                                       format_func=lambda x: st.session_state.class_names[x],
                                       index=int(st.session_state.modified_labels[idx]), key=f"fix_{idx}")
                    if new != st.session_state.modified_labels[idx]:
                        st.session_state.modified_labels[idx] = new
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("将所有疑似噪声改为预测标签"):
                    if st.session_state.predictions is not None:
                        noise_indices = np.where(st.session_state.sample_types == 2)[0]
                        if len(noise_indices) > 0:
                            modified_count = 0
                            for idx in noise_indices:
                                if idx < len(st.session_state.modified_labels) and idx < len(st.session_state.predictions):
                                    old_label = st.session_state.modified_labels[idx]
                                    new_label = st.session_state.predictions[idx]
                                    st.session_state.modified_labels[idx] = new_label
                                    if old_label != new_label:
                                        modified_count += 1
                            st.toast(f"✅ 已将 {modified_count} 个疑似噪声样本改为预测标签", icon="✅")
                        else:
                            st.warning("没有疑似噪声样本")
                    else:
                        st.warning("预测标签为空，请先运行检测")
            with col2:
                if st.button("重置所有修改"):
                    st.session_state.modified_labels = st.session_state.labels.copy()
                    st.toast("✅ 已重置所有修改", icon="🔄")

            st.markdown("---")
            st.subheader("📥 导出分析结果")
            if st.button("生成导出文件", key="export_btn"):
                sample_ids = [s['id'] for s in st.session_state.samples]
                n_samples = len(sample_ids)
                if st.session_state.true_labels is not None:
                    true_labels_str = [st.session_state.class_names[l] for l in st.session_state.true_labels[:n_samples]]
                    is_noise = [st.session_state.labels[i] != st.session_state.true_labels[i] for i in range(n_samples)]
                else:
                    true_labels_str = [None] * n_samples
                    is_noise = [False] * n_samples
                current_labels = [st.session_state.class_names[l] for l in st.session_state.labels[:n_samples]]
                modified_labels = [st.session_state.class_names[l] for l in st.session_state.modified_labels[:n_samples]]
                if st.session_state.predictions is not None:
                    pred_labels = [st.session_state.class_names[p] for p in st.session_state.predictions[:n_samples]]
                    pred_correct = [pred_labels[i] == true_labels_str[i] for i in range(n_samples)]
                    pred_correct_str = ['是' if x else '否' for x in pred_correct]
                else:
                    pred_labels = [None] * n_samples
                    pred_correct_str = [None] * n_samples
                if st.session_state.confidences is not None:
                    confs = st.session_state.confidences[:n_samples]
                else:
                    confs = [None] * n_samples
                if st.session_state.sample_types is not None:
                    sample_types_str = [['高可信', '低可信', '疑似噪声'][t] for t in st.session_state.sample_types[:n_samples]]
                else:
                    sample_types_str = [None] * n_samples
                if st.session_state.true_labels is not None:
                    correction_correct = [st.session_state.modified_labels[i] == st.session_state.true_labels[i] for i in range(n_samples)]
                else:
                    correction_correct = [False] * n_samples
                df = pd.DataFrame({
                    '样本ID': sample_ids,
                    '真实标签': true_labels_str,
                    '原始噪声标签': current_labels,
                    '修改后标签': modified_labels,
                    '预测标签': pred_labels,
                    '预测是否正确': pred_correct_str,
                    '置信度': confs,
                    '样本类型': sample_types_str,
                    '原始是否为噪声': ['是' if x else '否' for x in is_noise],
                    '修改是否正确': ['是' if x else '否' for x in correction_correct]
                })
                st.session_state.export_df = df
                st.success(f"✅ 已生成 {len(df)} 条记录，请点击下方按钮下载")
            if st.session_state.get('export_df') is not None:
                csv = st.session_state.export_df.to_csv(index=False, encoding='utf-8-sig').encode()
                st.download_button(
                    "📥 下载CSV文件",
                    csv,
                    f"analysis_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="download_csv"
                )
        else:
            st.info("请先运行检测")


if __name__ == "__main__":
    main()
