
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
import os

import cv2

from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
from single_test import analyze_knocks


def merge_small_clusters(original_labels, features, min_size=50):
    """
    合并样本数小于min_size的簇，直到所有簇都满足最小大小要求
    :param original_labels: 初始聚类标签
    :param features: 特征向量
    :param min_size: 最小簇大小
    :return: 合并后的标签数组
    """
    # 创建标签的副本，避免修改原始数据
    labels = np.copy(original_labels)

    # 统计每个簇的大小
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # 获取小簇列表（排除噪声点-1）
    small_clusters = [label for label, size in cluster_sizes.items() if size < min_size]

    # 循环合并小簇直到所有簇都满足大小要求
    while small_clusters:
        # 找到最近的两个小簇
        min_distance = float('inf')
        merge_pair = (None, None)

        # 检查所有小簇对
        for i, label1 in enumerate(small_clusters):
            for label2 in small_clusters[i + 1:]:
                # 获取两个簇的点集
                mask1 = (labels == label1)
                mask2 = (labels == label2)
                cluster1 = features[mask1]
                cluster2 = features[mask2]

                # 计算簇间最小距离
                dist_matrix = pairwise_distances(cluster1, cluster2)
                min_dist = np.min(dist_matrix)

                # 更新最小距离对
                if min_dist < min_distance:
                    min_distance = min_dist
                    merge_pair = (label1, label2)

        # 如果没有找到可合并的对（只剩一个孤立的簇）
        if min_distance == float('inf'):
            # 将孤立小簇转为噪声点
            for label in small_clusters:
                labels[labels == label] = -1
            break

        # 执行合并：将label2的所有点标记为label1
        label1, label2 = merge_pair
        labels[labels == label2] = label1

        # 更新簇大小信息
        new_size = cluster_sizes[label1] + cluster_sizes[label2]
        cluster_sizes[label1] = new_size

        # 删除被合并的簇
        del cluster_sizes[label2]
        small_clusters.remove(label2)

        # 检查合并后的簇是否仍然小于min_size
        if new_size < min_size:
            # 更新大小后保持在小簇列表中
            pass
        else:
            # 如果满足大小要求则移出小簇列表
            small_clusters.remove(label1)

    return labels


from sklearn.mixture import GaussianMixture


def perform_clustering(encoded_features, method):
    """
    执行非线性聚类，支持DBSCAN、谱聚类、KMeans和高斯混合模型
    :param encoded_features: 编码后的特征向量
    :param method: 聚类方法 ('dbscan', 'spectral', 'kmeans' 或 'gmm')
    :return: 聚类标签, 最佳聚类数
    """

    def relabel_by_size(labels):
        """按簇大小重新排序标签（从大到小），噪声点(-1)保持不变"""
        # 分离噪声点和有效聚类点
        noise_mask = (labels == -1)
        valid_labels = labels[~noise_mask]

        if len(valid_labels) == 0:
            return labels  # 只有噪声点时直接返回

        # 计算每个簇的大小
        unique_labels, counts = np.unique(valid_labels, return_counts=True)

        # 按簇大小降序排序
        size_order = np.argsort(-counts)
        sorted_labels = unique_labels[size_order]

        # 创建映射字典：旧标签 -> 新标签 (从0开始)
        label_mapping = {old: new for new, old in enumerate(sorted_labels)}

        # 创建新标签数组
        new_labels = labels.copy()
        for old_label in unique_labels:
            mask = (labels == old_label) & ~noise_mask
            new_labels[mask] = label_mapping[old_label]

        return new_labels

        # DBSCAN方法（适合密集核心+离散分布）

    if method == 'dbscan':
        # 自动确定eps参数：使用第5近邻距离的95%分位数
        nn = NearestNeighbors(n_neighbors=8)
        nn.fit(encoded_features)
        distances, _ = nn.kneighbors(encoded_features)
        k_distances = np.sort(distances[:, -1])
        eps = np.percentile(k_distances, 94)

        # 执行DBSCAN聚类
        db = DBSCAN(eps=eps, min_samples=4).fit(encoded_features)
        labels = db.labels_
        merged_labels = merge_small_clusters(labels, encoded_features)

        # 处理噪声点（标记为-1的转为单独一类）
        unique_labels = np.unique(merged_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        print(f"DBSCAN聚类数: {n_clusters} (含噪声点类)")

        return merged_labels, n_clusters

        # 谱聚类方法（使用RBF核处理非线性）
    elif method == 'spectral':
        best_score = -1
        best_labels = None
        best_n = 2

        for n_clusters in range(8, 9):
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='rbf',  # RBF核处理非线性
                gamma=0.1,  # 核参数
                random_state=42
            )
            labels = sc.fit_predict(encoded_features)

            # 跳过无效聚类
            if len(np.unique(labels)) < 2:
                continue

            # 计算轮廓系数
            score = silhouette_score(encoded_features, labels)
            print(f"n_clusters={n_clusters}: 轮廓系数={score:.4f}")

            if score > best_score:
                best_score = score
                best_labels = labels
                best_n = n_clusters

        print(f"最佳聚类数: {best_n} (轮廓系数={best_score:.4f})")
        return best_labels, best_n

    # KMeans方法
    elif method == 'KMeans':
        final_kmeans = KMeans(n_clusters=4, random_state=42)
        final_labels = final_kmeans.fit_predict(encoded_features)

        # 重新排序标签
        ordered_labels = relabel_by_size(final_labels)
        return ordered_labels, 4

    # 高斯混合模型方法
    elif method == 'gmm':

        best_n = 2


        gmm = GaussianMixture(
                n_components=best_n,
                random_state=42
        )
        labels = gmm.fit_predict(encoded_features)


        # 重新排序标签
        ordered_labels = relabel_by_size(labels)

        return ordered_labels, best_n







def plot_features(features_df,path):
    """
    绘制敲击声特征的三维可视化图
    横坐标: decay_time
    纵坐标: peak amplitude
    颜色映射: cluster结果


    """
    # 从DataFrame中提取数据
    mid_freq_ratios = features_df['mid_freq_ratio'].values
    rms_energys  = features_df['rms_energy'].values
    clusters = features_df['cluster'].values

    # 获取唯一的聚类数量
    unique_clusters, counts = np.unique(clusters, return_counts=True)




    cluster_colors = ['green', 'blue','yellow','red', 'purple', 'cyan', 'orange','magenta','brown','pink','black','white', 'gray', 'lime', 'gold', 'silver', 'navy', 'teal', 'coral',
    'indigo', 'violet', 'maroon', 'turquoise', 'olive', 'lavender', 'salmon',
    'khaki', 'plum', 'skyblue', 'orchid', 'crimson', 'sienna', 'tan', 'beige',
    'azure', 'ivory', 'chartreuse', 'tomato', 'slategray', 'seagreen', 'peru',
    'darkviolet', 'dodgerblue', 'firebrick', 'royalblue', 'steelblue', 'palegreen',
    'darkorange', 'lightcoral', 'mediumpurple']







    # 为每个点分配颜色
    point_colors = [cluster_colors[cluster] for cluster in clusters]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制散点图
    scatter = ax.scatter(
        mid_freq_ratios,  # X轴: 衰减时间
        rms_energys,  # Y轴: 频谱质心
        c=point_colors,  # 颜色: 聚类结果
        alpha=0.4,
        s=30,  # 点的大小
        edgecolor='w',
        linewidth=0.8
    )

    # 创建图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors[i], markersize=10,
                   label=f'Cluster {c}')
        for i, c in enumerate(unique_clusters)
    ]

    ax.legend(handles=legend_elements, loc='best', title='Clusters')

    # 设置标签和标题
    ax.set_xlabel('mid_freq_ratios', fontsize=12)
    ax.set_ylabel('RMS Energy ', fontsize=12)
    ax.set_title('Knock Analysis: Mid-freq Ratios vs RMS Energy', fontsize=14)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局

    save_path = str(
        path+'/dim.png')  # Replace with your desired path
    plt.savefig(save_path)
    plt.show()




def print_cluster_descriptions(df):
    """打印各聚类特征描述"""
    features = ['duration', 'decay_time', 'spectral_centroid',
                'low_freq_ratio', 'peak_amplitude','high_freq_ratio','mid_freq_ratio','rms_energy','dominant_frequency']

    cluster_means = df.groupby('cluster')[features].mean()


    print("\n=== 各聚类特征描述 ===")
    for cluster in cluster_means.index:
        print(f"\n◆ 聚类 {cluster} 特征:")

        print(f"- 平均衰减时间: {cluster_means.loc[cluster, 'decay_time']:.3f}s")
        print(f"- 平均频谱质心: {cluster_means.loc[cluster, 'spectral_centroid']:.1f}Hz")
        print(f"- 低频能量比例: {cluster_means.loc[cluster, 'low_freq_ratio']:.1%}")
        print(f"- 平均峰值振幅: {cluster_means.loc[cluster, 'peak_amplitude']:.3f}")
        print(f"- 高频能量比例: {cluster_means.loc[cluster, 'high_freq_ratio']:.1%}")
        print(f"- 中频能量比例: {cluster_means.loc[cluster, 'mid_freq_ratio']:.1%}")
        print(f"- RMS: {cluster_means.loc[cluster, 'rms_energy']:.3f}")
        print(f"- 音量峰值: {cluster_means.loc[cluster, 'peak_amplitude']:.3f}")
        print(f"- 基础频率: {cluster_means.loc[cluster, 'dominant_frequency']:.3f}")

        # 判断聚类类型



def plot_cluster_heatmap(df_all, wav_file_list, best_n, save,big=False):
    """绘制聚类结果的热力图（每个文件一列，每行代表一次敲击）"""
    # 获取所有文件的basename列表（按原始顺序）
    file_basenames = [os.path.basename(f) for f in wav_file_list]
    if big:
        df = df_all[df_all['file'].isin(file_basenames)]
    else:
        df=df_all

    # 获取所有文件中的最大敲击次数
    max_knocks = df.groupby('file')['knock_number'].max().max()

    # 创建聚类矩阵 (最大敲击次数 × 文件数)
    cluster_matrix = np.full((max_knocks, len(wav_file_list)), np.nan)

    # 填充聚类数据
    for file_idx, wav_file in enumerate(wav_file_list):
        file_basename = os.path.basename(wav_file)
        file_knocks = df[df['file'] == file_basename]

        if not file_knocks.empty:
            # 按敲击顺序排序
            file_knocks = file_knocks.sort_values('knock_number')
            for _, row in file_knocks.iterrows():
                knock_idx = row['knock_number'] - 1  # 转换为0-based索引
                cluster_matrix[knock_idx, file_idx] = row['cluster']

    cluster_counts = df['cluster'].value_counts()



    # 预定义颜色列表（确保绿色分配给最多的聚类）
    available_colors = ['green', 'blue','yellow','red', 'purple', 'cyan', 'orange','magenta','brown','pink','black','white', 'gray', 'lime', 'gold', 'silver', 'navy', 'teal', 'coral',
    'indigo', 'violet', 'maroon', 'turquoise', 'olive', 'lavender', 'salmon',
    'khaki', 'plum', 'skyblue', 'orchid', 'crimson', 'sienna', 'tan', 'beige',
    'azure', 'ivory', 'chartreuse', 'tomato', 'slategray', 'seagreen', 'peru',
    'darkviolet', 'dodgerblue', 'firebrick', 'royalblue', 'steelblue', 'palegreen',
    'darkorange', 'lightcoral', 'mediumpurple']



    # 创建颜色列表（按聚类编号排序）
    cluster_colors = [available_colors[i] for i in range(best_n)]
    cmap = plt.cm.colors.ListedColormap(cluster_colors[:best_n])

    # 设置NaN值的颜色（白色）
    cmap.set_bad(color='white')

    # 创建热力图
    plt.figure(figsize=(15, 10))

    # 使用imshow绘制热力图
    img = plt.imshow(cluster_matrix, cmap=cmap, aspect='auto',
                     interpolation='nearest', vmin=0, vmax=best_n - 1)

    # 添加颜色条
    cbar = plt.colorbar(img, ticks=np.arange(best_n))
    cbar.set_label('Cluster', fontsize=12)
    cbar.set_ticklabels([f'Cluster {i + 1}' for i in range(best_n)])

    # 设置坐标轴
    plt.ylabel('Knock Sequence', fontsize=12)
    plt.xlabel('Files', fontsize=12)

    # 设置文件标签（x轴）
    plt.xticks(np.arange(len(wav_file_list)), file_basenames, rotation=90, fontsize=8)

    # 设置敲击序列标签（y轴）
    plt.yticks(np.arange(max_knocks), np.arange(1, max_knocks + 1), fontsize=9)

    # 添加网格线
    plt.grid(visible=True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
    save_path = str(save + '/result.png')
    # 添加标题
    plt.title('Knock Sequence Cluster Heatmap', fontsize=14, pad=20)
    plt.savefig(
        save_path
    )
    plt.draw()  # 非阻塞显示(图形不会清空Z)
    plt.pause(1)  # 短暂暂停，确保窗口更新
    if hasattr(img, 'colorbar'):  # 检查是否存在 colorbar
        img.colorbar.remove()  # 移除 colorbar
    plt.gca().axis('off')  # Turn off all axes, labels, and ticks
    plt.title('')  # Remove title


    # --- Adjust margins to remove whitespace ---
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # --- Save only the central heatmap (no borders, no axes) ---
    save_path = str(save+'/plot.png')  # Replace with your desired path
    plt.savefig(
        save_path,
        bbox_inches='tight',  # Crop whitespace
        pad_inches=0,  # Remove padding
        transparent=True,  # Optional: Save with transparent background
        dpi=300  # Optional: Higher resolution
    )

    plt.close()  # Close the figure to free memory



if __name__ == "__main__":
    # 替换为你的文件夹路径
    wav_folder = "..."

    # 找到所有wav文件
    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]
    wav_files_sorted = sorted(wav_files)  # 按文件名排序

    wav_file_list = [os.path.join(wav_folder, f) for f in wav_files_sorted]

    print(f"共找到{len(wav_file_list)}个WAV文件。")

    # --------- 收集所有knock ----------
    all_features = []
    knock_file_map = []  # 记录knock属于哪个文件

    for wav_file_path in wav_file_list:
        print(f"\n====== 正在处理：{os.path.basename(wav_file_path)} ======")
        features, sample_rate, audio_data, peaks = analyze_knocks(wav_file_path)
        if not features:
            print("未检测到敲击，跳过。")
            continue
        for i, feat in enumerate(features):
            feat_copy = feat.copy()
            feat_copy['file'] = os.path.basename(wav_file_path)
            feat_copy['file_index'] = wav_file_list.index(wav_file_path)
            all_features.append(feat_copy)
            knock_file_map.append({'file': wav_file_path, 'file_index': wav_file_list.index(wav_file_path),
                                   'sample_rate': sample_rate, 'audio_data': audio_data, 'peaks': peaks, 'knock_index': i})

    print(f"共收集到{len(all_features)}个knock。")

    if not all_features:
        print("没有检测到任何knock，程序结束。")
        exit()

    # --------- 整体聚类 ----------
    df_all = pd.DataFrame(all_features)
    feature_cols = ['rms_energy','mid_freq_ratio']
    X = df_all[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    labels, best_n = perform_clustering(X_scaled,'...')

    df_all['cluster'] = labels


    # --- 汇总输出 ---
    print("\n每个文件内knock聚类分布：")
    print(df_all.groupby(['file', 'cluster']).size().unstack(fill_value=0))

    print("\n全部knock聚类分布：")
    print(df_all['cluster'].value_counts().sort_index())

    # --- 可视化整体聚类特征差异 ---

    plot_features(df_all,wav_folder)
    print("\n正在生成聚类热力图...")
    plot_cluster_heatmap(df_all, wav_file_list, best_n,save=wav_folder)

    building_img = cv2.imread('...')
    texture_img = cv2.imread(wav_folder+'/plot.png')

    # 定义墙面区域的四个角点（左上、右上、右下、左下）
    building_corners = np.array([[331, 109], [866, 101], [866, 211], [333, 216]], dtype=np.float32)

    # 定义目标图像的四个角点
    texture_corners = np.array([[0, 0], [texture_img.shape[1], 0],
                                [texture_img.shape[1], texture_img.shape[0]],
                                [0, texture_img.shape[0]]], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(texture_corners, building_corners)

    # 应用透视变换
    warped_texture = cv2.warpPerspective(texture_img, M, (building_img.shape[1], building_img.shape[0]))

    # 创建掩膜并融合
    mask = np.zeros_like(building_img)
    cv2.fillConvexPoly(mask, building_corners.astype(int), (255, 255, 255))
    mask = cv2.erode(mask, np.ones((0, 0), np.uint8))  # 缩小掩膜以平滑边缘
    center = (int((400 + 800) / 2), int((200 + 400) / 2))  # 计算墙面中心点并转为整数
    result = cv2.seamlessClone(
        warped_texture,
        building_img,
        mask,
        center,
        cv2.NORMAL_CLONE
    )
    # 定义目标图像的四个角点（直接取它的四个角）
    texture_corners = np.array([
        [0, 0],  # 左上
        [texture_img.shape[1], 0],  # 右上
        [texture_img.shape[1], texture_img.shape[0]],  # 右下
        [0, texture_img.shape[0]]  # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(texture_corners, building_corners)

    # 应用透视变换，使目标图像贴合墙面区域
    warped_texture = cv2.warpPerspective(
        texture_img,
        M,
        (building_img.shape[1], building_img.shape[0])  # 输出尺寸 = 建筑物图片尺寸 (1280×571)
    )

    # 创建掩膜（只保留墙面区域）
    mask = np.zeros_like(building_img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, building_corners.astype(int), (255, 255, 255))

    # 方法1：直接覆盖（无柔化边缘）
    result = building_img.copy()
    result[mask > 0] = warped_texture[mask > 0]  # 仅替换mask区域

    # 方法2：使用 bitwise 操作（效果相同）
    # result = cv2.bitwise_and(building_img, cv2.bitwise_not(mask))
    # result = cv2.bitwise_or(result, cv2.bitwise_and(warped_texture, mask))

    cv2.imwrite(wav_folder+'/dex.png', result)