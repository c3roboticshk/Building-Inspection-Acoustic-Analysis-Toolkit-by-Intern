import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf
import noisereduce as nr


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def remove_frequency(input_file, output_file, lowcut, highcut):
    try:
        # 读取音频
        data, fs = sf.read(input_file, dtype='float32')
        print(f"输入音频 - 长度: {len(data)}采样点, 范围: [{np.min(data):.4f}, {np.max(data):.4f}]")

        # 设计滤波器
        bandpass_data = filtfilt(*butter_bandpass(lowcut, highcut, fs), data)
        filtered_data = data - bandpass_data

        # 检查滤波结果
        print(f"滤波后 - 范围: [{np.min(filtered_data):.4f}, {np.max(filtered_data):.4f}]")

        # 确保数据有效
        if np.all(filtered_data == 0):
            raise ValueError("滤波后数据全为零！请检查滤波器参数")

        # 保存结果
        sf.write(output_file, filtered_data, fs)
        print("处理成功完成")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


# 测试参数
input_audio = '...'
output_audio = '...'
lowcut_freq =
highcut_freq =

remove_frequency(input_audio, output_audio, lowcut_freq, highcut_freq)