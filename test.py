import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np
from folder_test import perform_clustering
from single_test import analyze_knocks
from folder_test import perform_clustering, plot_features, print_cluster_descriptions, plot_cluster_heatmap








def collect_all_knocks(root_dir):
    """遍历所有墙面、文件，收集所有knock特征，附带墙面名和文件名"""
    wall_dirs = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
                 if os.path.isdir(os.path.join(root_dir, d))]
    all_features = []
    file_map = []  # (wall_name, wav_path)列表，保持顺序
    for wall_dir in wall_dirs:
        wall_name = os.path.basename(wall_dir)
        wav_files = sorted([f for f in os.listdir(wall_dir) if f.lower().endswith('.wav')])
        for wav_file in wav_files:
            wav_path = os.path.join(wall_dir, wav_file)
            features, _, _, _ = analyze_knocks(wav_path)
            for feat in features:
                feat_copy = feat.copy()
                feat_copy['file'] = wav_file
                feat_copy['wall'] = wall_name
                all_features.append(feat_copy)
            file_map.append((wall_name, wav_path))
    return all_features, file_map, wall_dirs


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





    labels, best_n = perform_clustering(X,method='gmm')
    df_all['cluster'] = labels

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