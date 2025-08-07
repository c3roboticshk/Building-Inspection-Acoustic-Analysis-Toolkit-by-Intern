import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks


def vad(signal, sr, threshold=0.05, min_silence=0.1, min_event=0.05):
    """
    声音活动检测（VAD）分割声音事件
    """
    # 计算信号能量包络
    envelope = np.abs(signal)
    window_size = int(sr * 0.01)  # 10ms窗口
    envelope = np.convolve(envelope, np.ones(window_size) / window_size, mode='same')

    # 动态阈值（最大能量的5%）
    dyn_threshold = max(threshold * np.max(envelope), 1e-5)

    # 检测有声段
    active = envelope > dyn_threshold
    active_diff = np.diff(active.astype(int))
    starts = np.where(active_diff == 1)[0]
    ends = np.where(active_diff == -1)[0]

    # 处理边界情况
    if starts.size == 0 or ends.size == 0:
        return []
    if ends[0] < starts[0]:
        ends = ends[1:]
    if starts[-1] > ends[-1]:
        starts = starts[:-1]

    # 合并间隔过近的事件
    min_silence_samples = int(min_silence * sr)
    min_event_samples = int(min_event * sr)
    events = []
    current_start = starts[0]

    for i in range(len(starts) - 1):
        if starts[i + 1] - ends[i] > min_silence_samples:
            if ends[i] - current_start > min_event_samples:
                events.append((current_start, ends[i]))
            current_start = starts[i + 1]

    # 添加最后一个事件
    if ends[-1] - current_start > min_event_samples:
        events.append((current_start, ends[-1]))

    return events


def calculate_delay(sig1, sig2, sr, max_delay_samples):
    """
    计算信号间的时间延迟（互相关方法）
    """
    # 归一化信号
    sig1 = (sig1 - np.mean(sig1)) / np.std(sig1)
    sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)

    # 计算互相关
    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig2) + 1, len(sig1))

    # 限制在最大延迟范围内搜索
    valid_indices = np.where(np.abs(lags) <= max_delay_samples)[0]
    if len(valid_indices) == 0:
        return None
    max_idx = valid_indices[np.argmax(corr[valid_indices])]

    # 转换为时间（秒）
    delay_seconds = lags[max_idx] / sr
    return delay_seconds


def main():
    # ================= 参数设置 =================
    mic1_file = 'mic1.wav'  # 麦克风1的音频文件
    mic2_file = 'mic2.wav'  # 麦克风2的音频文件
    d = 0.4  # 麦克风间距（米）
    t = 0.0011595  # 时间差阈值（秒），小于此值的事件将被过滤
    vad_threshold = 0.05  # VAD检测阈值（相对最大能量的比例）
    # ===========================================

    # 读取音频文件
    sr1, mic1 = wavfile.read(mic1_file)
    sr2, mic2 = wavfile.read(mic2_file)

    # 确保采样率相同
    if sr1 != sr2:
        raise ValueError("两个音频文件的采样率不同！")

    # 转换为单声道（如果是立体声）
    if mic1.ndim > 1:
        mic1 = mic1[:, 0]
    if mic2.ndim > 1:
        mic2 = mic2[:, 0]

    # 计算最大可能延迟（基于麦克风间距）
    speed_of_sound = 340.0  # 声速（m/s）
    max_delay = d / speed_of_sound
    max_delay_samples = int(max_delay * sr1)

    # 检测声音事件（使用麦克风1）
    events = vad(mic1, sr1, threshold=vad_threshold)

    # 处理每个声音事件
    valid_events = []
    for start, end in events:
        # 麦克风1的事件信号
        sig1 = mic1[start:end]

        # 麦克风2的搜索范围（扩展最大延迟）
        search_start = max(0, start - max_delay_samples)
        search_end = min(len(mic2), end + max_delay_samples)
        sig2 = mic2[search_start:search_end]

        # 计算时间延迟
        delay = calculate_delay(sig1, sig2, sr1, max_delay_samples)

        if delay is None:
            continue

        # 只处理1号麦克风先到达的延迟（正值）
        if delay < 0:
            continue

        # 过滤时间差小于阈值的事件
        if delay >= t:
            valid_events.append((start / sr1, end / sr1, delay))

    # 输出结果
    print(f"检测到 {len(events)} 个声音事件")
    print(f"保留 {len(valid_events)} 个时间差 ≥ {t:.6f} 秒的事件")
    print("\n有效事件详情 [开始时间(秒), 结束时间(秒), 延迟(秒)]:")
    for event in valid_events:
        print(f"{event[0]:.4f}\t{event[1]:.4f}\t{event[2]:.6f}")


if __name__ == "__main__":
    main()