import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from folder_test import analyze_knocks, plot_cluster_heatmap, plot_features


# 新增函数：计算固定长度的FFT
def compute_fft(segment, n_fft=2048):
    """计算固定长度的FFT频谱"""
    if len(segment) < n_fft:
        padded_segment = np.pad(segment, (0, n_fft - len(segment)), 'constant')
    else:
        padded_segment = segment[:n_fft]
    fft = np.fft.rfft(padded_segment)
    return np.abs(fft)


# 修改后的频谱相似度函数
def spectral_similarity(spectrum1, spectrum2):
    """计算两个频谱之间的余弦相似度"""
    # 确保输入是1D向量
    spectrum1 = np.asarray(spectrum1).flatten()
    spectrum2 = np.asarray(spectrum2).flatten()

    # 检查形状是否匹配
    if spectrum1.shape != spectrum2.shape:
        min_len = min(len(spectrum1), len(spectrum2))
        spectrum1 = spectrum1[:min_len]
        spectrum2 = spectrum2[:min_len]

    dot_product = np.dot(spectrum1, spectrum2)
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    return dot_product / (norm1 * norm2 + 1e-10)  # 防止除以零


def create_model(input_dim):
    """创建神经网络模型（根据图中结构修改）"""
    model = Sequential([
        Dense(8, activation='relu', input_shape=(input_dim,)),  # 第一隐藏层
        Dropout(0.2),
        Dense(4, activation='relu'),  # 第二隐藏层
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(good_file, bad_file, model_path='knock_detection_model.h5'):
    """使用合格和不合格样本训练模型（添加频谱奖励）"""
    # 分析合格样本
    good_features, good_knocks, _, _ = analyze_knocks(good_file)
    good_df = pd.DataFrame(good_features)
    good_df['label'] = 1  # 合格标签

    # 分析不合格样本
    bad_features, bad_knocks, _, _ = analyze_knocks(bad_file)
    bad_df = pd.DataFrame(bad_features)
    bad_df['label'] = 0  # 不合格标签

    # 计算参考频谱 (使用固定长度FFT)
    good_spectra = [compute_fft(feature['segment']) for feature in good_features]
    bad_spectra = [compute_fft(feature['segment']) for feature in bad_features]

    if not good_spectra or not bad_spectra:
        raise ValueError("没有检测到敲击事件，无法计算参考频谱")

    good_ref_spectrum = np.mean(good_spectra, axis=0)
    bad_ref_spectrum = np.mean(bad_spectra, axis=0)

    # 合并数据集
    df = pd.concat([good_df, bad_df], ignore_index=True)

    # ====== 特征工程（根据图中的输入特征） ======
    # 原始特征
    df['X1'] = df['rms_energy']  # 假设X1对应RMS能量
    df['X2'] = df['mid_freq_ratio']  # 假设X2对应中频比例

    # 添加新特征（根据图中：X₃², X₄², sin(X₁), sin(X₂)
    df['X3_squared'] = df['low_freq_ratio'] ** 2
    df['X4_squared'] = df['high_freq_ratio'] ** 2
    df['sin_X1'] = np.sin(df['X1'])
    df['sin_X2'] = np.sin(df['X2'])

    # 最终特征列
    feature_cols = ['X1', 'X2', 'X3_squared', 'X4_squared', 'sin_X1', 'sin_X2']

    # 准备特征和标签
    X = df[feature_cols]
    y = df['label']

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集（图中比例50%）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=42
    )

    # 创建模型
    model = create_model(X_train.shape[1])

    # 自定义优化器配置
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型（批量大小改为10，如图中所示）
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=10,  # 图中batch size=10
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n模型评估结果 - 测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")

    # 保存模型和预处理对象
    model.save(model_path)
    np.save('knock_scaler.npy', scaler)

    # 保存参考频谱 (确保是1D数组)
    np.save('good_ref_spectrum.npy', good_ref_spectrum.flatten())
    np.save('bad_ref_spectrum.npy', bad_ref_spectrum.flatten())

    # 绘制训练历史
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model_Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Duration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Duration')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return model, scaler, good_ref_spectrum, bad_ref_spectrum


def get_knock_spectra_from_results(results):
    """从分析结果中获取频谱 (使用固定长度FFT)"""
    spectra = []
    for r in results:
        segment = r['segment']
        spectrum = compute_fft(segment)
        spectra.append(spectrum)
    return spectra


def analyze_folder(folder_path, good_file, bad_file):
    """分析文件夹中的所有WAV文件（添加频谱奖励）"""
    # 训练或加载模型
    model_path = 'knock_detection_model.h5'
    scaler_path = 'knock_scaler.npy'
    good_spectrum_path = 'good_ref_spectrum.npy'
    bad_spectrum_path = 'bad_ref_spectrum.npy'

    if all(os.path.exists(p) for p in [model_path, scaler_path, good_spectrum_path, bad_spectrum_path]):
        print("加载预训练模型和频谱参考...")
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
        good_ref_spectrum = np.load(good_spectrum_path).flatten()  # 确保是1D
        bad_ref_spectrum = np.load(bad_spectrum_path).flatten()  # 确保是1D
    else:
        print("训练新模型...")
        model, scaler, good_ref_spectrum, bad_ref_spectrum = train_model(good_file, bad_file, model_path)

    # 获取文件夹中的所有WAV文件
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    results = []
    all_details = pd.DataFrame()

    # 特征列（与训练时一致）
    feature_cols = ['X1', 'X2', 'X3_squared', 'X4_squared', 'sin_X1', 'sin_X2']

    for file in wav_files:
        file_path = os.path.join(folder_path, file)

        features, sample_rate, audio_data, peaks = analyze_knocks(file_path)
        if not features:
            print(f"文件 {file} 中没有检测到敲击声")
            continue

        df = pd.DataFrame(features)
        spectra = get_knock_spectra_from_results(features)

        # 特征工程（与训练时一致）
        df['X1'] = df['rms_energy']
        df['X2'] = df['mid_freq_ratio']
        df['X3_squared'] = df['low_freq_ratio'] ** 2
        df['X4_squared'] = df['high_freq_ratio'] ** 2
        df['sin_X1'] = np.sin(df['X1'])
        df['sin_X2'] = np.sin(df['X2'])

        # 预处理特征
        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        # 预测
        predictions = model.predict(X_scaled)
        df['quality'] = (predictions > 0.5).astype(int)
        df['quality_prob'] = predictions

        # 计算频谱奖励
        spectral_rewards = []
        for spectrum in spectra:
            sim_good = spectral_similarity(spectrum, good_ref_spectrum)
            diff_bad = 1 - spectral_similarity(spectrum, bad_ref_spectrum)
            reward = float(sim_good + diff_bad)
            spectral_rewards.append(reward)

        df['spectral_reward'] = spectral_rewards

        # 调整质量评分（结合预测概率和频谱奖励）
        df['adjusted_quality'] = (0.7 * df['quality_prob'] + 0.3 * df['spectral_reward']).clip(0, 1)
        df['final_quality'] = (df['adjusted_quality'] > 0.5).astype(int)
        df['file'] = file
        df['cluster'] = df['final_quality']  # 使用质量作为聚类标签

        # 计算合格率（基于调整后的质量评分）
        quality_rate = df['final_quality'].mean()

        # 添加到结果
        results.append({
            'file': file,
            'total_knocks': len(df),
            'good_knocks': df['final_quality'].sum(),
            'quality_rate': quality_rate,
            'quality_status': '合格' if quality_rate > 0.5 else '不合格',
            'details': df
        })

        # 收集所有细节
        all_details = pd.concat([all_details, df])

    return results, all_details


# 主程序
if __name__ == "__main__":
    # 设置文件路径
    GOOD_FILE = "/Users/ywsun/Desktop/pythonProject/C3/audio_samples/no_fan_pc_usbc.wav"  # 合格样本
    BAD_FILE = "/Users/ywsun/Desktop/pythonProject/C3/audio_samples/hollow_no_fan_pc_usbc.wav"  # 不合格样本
    FOLDER_PATH = "..."
    dic=FOLDER_PATH# 待检测文件夹


    # 确保输出路径存在
    os.makedirs(FOLDER_PATH, exist_ok=True)

    # 确保样本文件存在
    if not os.path.exists(GOOD_FILE):
        print(f"错误: 合格样本文件 {GOOD_FILE} 不存在")
    elif not os.path.exists(BAD_FILE):
        print(f"错误: 不合格样本文件 {BAD_FILE} 不存在")
    elif not os.path.exists(FOLDER_PATH):
        print(f"错误: 检测文件夹 {FOLDER_PATH} 不存在")
    else:
        # 分析文件夹中的文件
        analysis_results, all_details = analyze_folder(FOLDER_PATH, GOOD_FILE, BAD_FILE)

        # 保存详细结果
        detail_path = os.path.join(FOLDER_PATH, 'knock_analysis_details.csv')
        all_details.to_csv(detail_path, index=False)
        print(f"\n详细分析结果已保存到 {detail_path}")

        # 获取文件列表
        wav_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith('.wav')]

        # 绘制热力图
        print("生成热力图...")
        plot_cluster_heatmap(
            all_details,
            wav_files,
            len(all_details['cluster'].unique()),
            dic

        )


        # 绘制特征散点图
        print("生成特征散点图...")
        plot_features(all_details,FOLDER_PATH)
