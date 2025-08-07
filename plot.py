import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def analyze_knocks(wav_file_path):
    """分析WAV文件中的敲击声"""
    # 加载WAV文件
    sample_rate, audio_data = wavfile.read(wav_file_path)

    # 转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 归一化音频数据
    audio_data = audio_data / np.max(np.abs(audio_data))

    # 检测敲击声
    peaks, _ = find_peaks(np.abs(audio_data), height=0.1, distance=sample_rate * 0.5)

    # 提取特征
    features = []
    for i, peak in enumerate(peaks):
        start = max(0, peak - int(0.1 * sample_rate))
        end = min(len(audio_data), peak + int(0.5 * sample_rate))
        knock_segment = audio_data[start:end]

        # 时域特征
        duration = len(knock_segment) / sample_rate
        rms_energy = np.sqrt(np.mean(knock_segment ** 2))
        peak_amplitude = np.max(np.abs(knock_segment))

        # 计算衰减时间
        threshold = 0.17 * peak_amplitude
        above_threshold = np.where(np.abs(knock_segment) > threshold)[0]
        decay_time = (above_threshold[-1] - above_threshold[0]) / sample_rate if len(above_threshold) > 0 else 0

        # 频域特征
        fft = np.fft.rfft(knock_segment)
        fft_freqs = np.fft.rfftfreq(len(knock_segment), 1 / sample_rate)
        fft_mag = np.abs(fft)

        # 频谱质心
        spectral_centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)

        # 频率比例
        low_freq_ratio = np.sum(fft_mag[fft_freqs < 500]) / np.sum(fft_mag)
        mid_freq_ratio = np.sum(fft_mag[(fft_freqs >= 500) & (fft_freqs < 1100)]) / np.sum(fft_mag)
        high_freq_ratio = np.sum(fft_mag[fft_freqs >= 1100]) / np.sum(fft_mag)

        features.append({
            'knock_number': i + 1,
            'duration': duration,
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'decay_time': decay_time,
            'spectral_centroid': spectral_centroid,
            'low_freq_ratio': low_freq_ratio,
            'mid_freq_ratio': mid_freq_ratio,
            'high_freq_ratio': high_freq_ratio,
            'dominant_frequency': fft_freqs[np.argmax(fft_mag)],
            'segment': knock_segment
        })

    return features, sample_rate, audio_data, peaks


def plot_features(features_df):
    """
    绘制敲击声特征的三维可视化图
    横坐标: decay_time (衰减时间)
    纵坐标: spectral_centroid (频谱质心)
    颜色映射: peak_amplitude (峰值幅度)

    参数:
        features_df (DataFrame): 包含所有敲击特征的DataFrame
    """
    # 从DataFrame中提取数据
    decay_times = features_df['mid_freq_ratio'].values
    rms_energys = features_df['rms_energy'].values
    mid_freq_ratios = features_df['decay_time'].values

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 7))

    # 使用新的方法获取颜色映射
    cmap = plt.colormaps['coolwarm']  # 或者使用 plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=min(mid_freq_ratios), vmax=max(mid_freq_ratios))

    # 绘制散点图，颜色表示峰值幅度
    scatter = ax.scatter(
        decay_times,  # X轴: 衰减时间
        rms_energys,  # Y轴: 频谱质心
        c=mid_freq_ratios,  # 颜色: 峰值幅度
        cmap=cmap,
        norm=norm,
        alpha=0.4,
        s=20,  # 点的大小
        edgecolor='none'
    )

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Decay time', fontsize=12)  # 修正为峰值幅度

    # 设置标签和标题
    ax.set_xlabel('mid_freq_ratio', fontsize=12)  # X轴标签改为衰减时间
    ax.set_ylabel('RMS Energy', fontsize=12)  # Y轴标签改为频谱质心
    ax.set_title('Knock Analysis: Decay Time vs Spectral Centroid\n(Color indicates Peak Amplitude)', fontsize=14)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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
                                   'sample_rate': sample_rate, 'audio_data': audio_data, 'peaks': peaks,
                                   'knock_index': i})

    print(f"共收集到{len(all_features)}个knock。")

    if not all_features:
        print("没有检测到任何knock，程序结束。")
        exit()

    # 创建DataFrame
    df_all = pd.DataFrame(all_features)

    # 绘制特征图
    plot_features(df_all)