import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import os


def analyze_knocks(wav_file_path, save_dir=None):
    sample_rate, audio_data = wavfile.read(wav_file_path)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Find the knocking sounds (peaks in amplitude)
    peaks, _ = find_peaks(np.abs(audio_data), height=0.05, distance=sample_rate * 0.5)

    # Filter peaks to ensure minimum 0.9 seconds between knocks
    filtered_peaks = []
    if len(peaks) > 0:
        filtered_peaks.append(peaks[0])  # Always keep the first peak
        for i in range(1, len(peaks)):
            time_diff = (peaks[i] - filtered_peaks[-1]) / sample_rate
            if time_diff >= 0.9:
                filtered_peaks.append(peaks[i])

    # Use the filtered peaks for further processing
    peaks = np.array(filtered_peaks)

    print(f"found {len(peaks)}")

    # Prepare save directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Analyze each knock
    results = []
    for i, peak in enumerate(peaks):
        # Extract the knock segment (100ms before to 500ms after the peak)
        start = max(0, peak - int(0.1 * sample_rate))
        end = min(len(audio_data), peak + int(0.5 * sample_rate))
        knock_segment = audio_data[start:end]

        # 保存每一个 knock_segment 为单独的 wav 文件
        if save_dir is not None:
            # 恢复float归一化为int16
            knock_segment_int16 = np.int16(knock_segment * 32767)
            save_path = os.path.join(save_dir, f"knock_{i + 1}.wav")
            wavfile.write(save_path, sample_rate, knock_segment_int16)
            print(f"Saved: {save_path}")

        # Calculate time domain features
        duration = len(knock_segment) / sample_rate
        rms_energy = np.sqrt(np.mean(knock_segment ** 2))

        # Calculate frequency domain features
        fft = np.fft.rfft(knock_segment)
        fft_freqs = np.fft.rfftfreq(len(knock_segment), 1 / sample_rate)
        fft_mag = np.abs(fft)



        # Frequency content ratios
        low_freq_mask = fft_freqs < 500  # Frequencies below 1kHz
        mid_freq_mask = (fft_freqs >= 500) & (fft_freqs < 1100)
        high_freq_mask = fft_freqs >= 1100

        low_freq_ratio = np.sum(fft_mag[low_freq_mask]) / np.sum(fft_mag)
        mid_freq_ratio = np.sum(fft_mag[mid_freq_mask]) / np.sum(fft_mag)
        high_freq_ratio = np.sum(fft_mag[high_freq_mask]) / np.sum(fft_mag)

        # Decay time (time for amplitude to drop to 10% of peak)
        peak_amplitude = np.max(np.abs(knock_segment))
        threshold = 0.2 * peak_amplitude
        above_threshold = np.where(np.abs(knock_segment) > threshold)[0]
        if len(above_threshold) > 0:
            decay_time = (above_threshold[-1] - above_threshold[0]) / sample_rate
        else:
            decay_time = 0

        # 频谱质心
        spectral_centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)


        # Store results
        results.append({
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

    return results, sample_rate, audio_data, peaks


def plot_results(results, sample_rate, audio_data, peaks):
    # Plot the waveform with knock markers
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(audio_data)) / sample_rate
    plt.plot(time_axis, audio_data)
    plt.plot(peaks / sample_rate, audio_data[peaks], 'ro', label='Knocks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform with Detected Knocks')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot individual knocks and their features
    for i, result in enumerate(results):
        plt.figure(figsize=(12, 8))

        # Plot waveform
        plt.subplot(2, 1, 1)
        start = max(0, peaks[i] - int(0.1 * sample_rate))
        end = min(len(audio_data), peaks[i] + int(0.5 * sample_rate))
        knock_segment = audio_data[start:end]
        time_axis = np.arange(len(knock_segment)) / sample_rate
        plt.plot(time_axis, knock_segment)
        plt.title(f'Knock {i + 1} ')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        # Plot frequency spectrum
        plt.subplot(2, 1, 2)
        fft = np.fft.rfft(knock_segment)
        fft_freqs = np.fft.rfftfreq(len(knock_segment), 1 / sample_rate)
        fft_mag = np.abs(fft)
        plt.plot(fft_freqs, fft_mag)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 5000)
        major_ticks = np.arange(0, 5001, 200)  # 主刻度每500Hz
        minor_ticks = np.arange(0, 5001, 100)  # 次刻度每100Hz

        plt.xticks(major_ticks)  # 设置主刻度
        plt.gca().set_xticks(minor_ticks, minor=True)  # 设置次刻度

        plt.ylim(0, max(fft_mag) + 300)
        plt.grid()

        plt.tight_layout()
        plt.show()

        # Print results
        print(f"\nKnock {i + 1} Analysis:")
        print(f"- Duration: {result['duration']:.3f} s")
        print(f"- RMS Energy: {result['rms_energy']:.4f}")
        print(f"- Dominant Frequency: {result['dominant_frequency']:.1f} Hz")
        print(
            f"- Frequency Ratios - Low: {result['low_freq_ratio']:.2%}, Mid: {result['mid_freq_ratio']:.2%}, High: {result['high_freq_ratio']:.2%}")
        print(f"- Decay Time: {result['decay_time']:.3f} s")



        # 打印所有幅度大于500的频率
        significant_freqs = []
        for j, mag in enumerate(fft_mag):
            if mag > 500:
                significant_freqs.append((fft_freqs[j], mag))

        if significant_freqs:
            print("\nFrequencies with magnitude > 500:")
            for freq, mag in significant_freqs:
                print(f"  Frequency: {freq:.1f} Hz, Magnitude: {mag:.1f}")
        else:
            print("\nNo frequencies with magnitude > 500")


if __name__ == "__main__":
    # Replace with your WAV file path
    wav_file_path = (
        '...')
    knock_save_dir = "..."
    # Analyze the WAV file and save knock segments
    results, sample_rate, audio_data, peaks = analyze_knocks(wav_file_path, save_dir=None)
    # Plot and display results
    plot_results(results, sample_rate, audio_data, peaks)
