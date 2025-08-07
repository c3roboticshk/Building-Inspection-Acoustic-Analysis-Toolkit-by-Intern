
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from single_test import analyze_knocks
from folder_test import perform_clustering, plot_features, print_cluster_descriptions, plot_cluster_heatmap
from test import collect_all_knocks
from sklearn.mixture import GaussianMixture


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




def main():
    root_dir = "..."

    all_features, file_map, wall_dirs = collect_all_knocks(root_dir)
    if not all_features:
        print("没有检测到任何knock，退出。")
        return

    # 全部knock整体聚类
    df_all = pd.DataFrame(all_features)

    feature_cols = [
                     'mid_freq_ratio', 'rms_energy']
    X = df_all[feature_cols].values

    dim=2

    encoded_features = extract_features_with_autoencoder(X, latent_dim=dim)

    # 使用降维后的特征进行聚类
    labels, best_n = perform_clustering(encoded_features,method='dbscan')
    df_all['cluster'] = labels


    score = silhouette_score(encoded_features, labels)
    print(f"维度{dim}的轮廓系数: {score:.4f}")

    print("\n每个文件内knock聚类分布：")
    print(df_all.groupby(['file', 'cluster']).size().unstack(fill_value=0))

    print("\n全部knock聚类分布：")
    print(df_all['cluster'].value_counts().sort_index())

    print_cluster_descriptions(df_all)
    plot_features(df_all,root_dir)



    # 存储每个墙面的热力图路径
    wall_textures = {}

    # 按文件夹名顺序处理每个墙面
    for wall_dir in wall_dirs:
        wall_name = os.path.basename(wall_dir)
        # 取出该墙面所有knock数据
        df_wall = df_all[df_all['wall'] == wall_name].copy()
        # 可视化聚类特征差异

        # 找到该墙面所有wav文件路径（按file_map顺序筛选）
        wav_file_list = [wav_path for w, wav_path in file_map if w == wall_name]
        if len(df_wall) > 0 and wav_file_list:
            # 生成热力图，保存路径
            plot_path = plot_cluster_heatmap(df_all, wav_file_list, best_n, wall_dir,big=True)
            wall_textures[wall_name] = plot_path
        else:
            print(f"墙面{wall_name}没有有效knock或wav文件，跳过热力图。")



if __name__ == "__main__":
    main()