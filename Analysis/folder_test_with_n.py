import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from single_test import analyze_knocks
from folder_test import perform_clustering, plot_features, print_cluster_descriptions, plot_cluster_heatmap


def build_autoencoder(input_dim, latent_dim=8):
    # 定义网络输入层：指定输入特征的维度
    input_layer = Input(shape=(input_dim,))

    # ===== 编码器部分 =====
    # 第一编码层：64个神经元，ReLU激活函数
    encoded = Dense(64, activation='relu')(input_layer)
    # 第二编码层：32个神经元，ReLU激活函数
    encoded = Dense(32, activation='relu')(encoded)
    # 潜在空间层：核心降维层（默认8维），ReLU激活
    encoded = Dense(latent_dim, activation='relu')(encoded)

    # ===== 解码器部分 =====
    # 第一解码层：32个神经元，ReLU激活（与编码器对称）
    decoded = Dense(32, activation='relu')(encoded)
    # 第二解码层：64个神经元，ReLU激活
    decoded = Dense(64, activation='relu')(decoded)
    # 输出层：维度与输入相同，线性激活（重建原始输入）
    decoded = Dense(input_dim, activation='linear')(decoded)

    # ===== 模型构建 =====
    # 创建自编码器模型：输入→完整编码解码流程
    autoencoder = Model(input_layer, decoded)
    # 创建编码器模型：仅保留编码部分（用于特征提取）
    encoder = Model(input_layer, encoded)

    # ===== 模型编译 =====
    # 使用Adam优化器（学习率0.001）
    # 损失函数：均方误差（衡量重建质量）
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')

    return autoencoder, encoder

def extract_features_with_autoencoder(features, latent_dim):
    """使用自编码器提取特征"""
    # 标准化输入特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 构建自编码器
    autoencoder, encoder = build_autoencoder(
        input_dim=X_scaled.shape[1],
        latent_dim=latent_dim
    )

    # 训练自编码器
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=150,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        verbose=0
    )

    # 检测过拟合迹象
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    overfit_threshold = 0.2  # 允许的差异阈值

    if min(val_loss) > min(train_loss) * (1 + overfit_threshold):
        print(f"警告：检测到过拟合 (训练损失:{min(train_loss):.4f}, 验证损失:{min(val_loss):.4f})")

    # 提取潜在特征
    encoded_features = encoder.predict(X_scaled)
    return encoded_features





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

    feature_cols = [
                     'mid_freq_ratio', 'rms_energy']
    X = df_all[feature_cols].values


    encoded_features = extract_features_with_autoencoder(X, 2)

    # 使用降维后的特征进行聚类
    labels, best_n = perform_clustering(encoded_features,'...')
    df_all['cluster'] = labels



    # --- 汇总输出 ---
    print("\n每个文件内knock聚类分布：")
    print(df_all.groupby(['file', 'cluster']).size().unstack(fill_value=0))

    print("\n全部knock聚类分布：")
    print(df_all['cluster'].value_counts().sort_index())

    # --- 可视化整体聚类特征差异 ---
    #visualize_cluster_differences(df_all)
    plot_features(df_all,wav_folder)


    print("\n正在生成聚类热力图...")
    plot_cluster_heatmap(df_all, wav_file_list, best_n,save=wav_folder)

    building_img = cv2.imread('...')
    texture_img = cv2.imread(
        wav_folder+'/plot.png')

    # 定义墙面区域的四个角点（左上、右上、右下、左下）
    building_corners = np.array([
        [331, 109], [866, 101], [866, 211], [333, 216]
    ], dtype=np.float32)

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