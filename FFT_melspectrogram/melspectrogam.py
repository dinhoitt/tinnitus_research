import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
import librosa
import librosa.display
from scipy.fft import fft, ifft
from pydub import AudioSegment

# wav file 로딩
audio = AudioSegment.from_file(r"C:\Users\taeso\Desktop\testfft.wav")

# wav 파일 로딩
fs, data = wavfile.read("C:/Users/taeso/Desktop/testfft.wav")

# 데이터를 Float 타입으로 변환
data_float = data.astype(np.float64)

# 데이터의 최대 절대값 찾기
max_val = np.max(np.abs(data_float))

# 신호를 최대 절대값으로 나누어 정규화
norm_data = data_float / max_val

# 시간 축을 생성
time = np.arange(0, len(data)) / fs

# 전체 플롯을 위한 Figure 생성
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # 3개의 행과 1개의 열

# 첫 번째 서브플롯: 시간 도메인 신호
ax1 = fig.add_subplot(gs[0])
ax1.plot(time, norm_data)
ax1.set_title('Normalized Signal in Time Domain')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

# 두 번째 서브플롯: 스펙트로그램
ax2 = fig.add_subplot(gs[1])
frequencies, times, Sxx = spectrogram(data, fs, nperseg=1024)
img = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec]')
ax2.set_title('Spectrogram')
fig.colorbar(img, ax=ax2, format='%+2.0f dB')

# 세 번째 서브플롯: 멜스펙트로그램
ax3 = fig.add_subplot(gs[2])
S = librosa.feature.melspectrogram(y=norm_data, sr=fs, n_mels=128, fmax=fs/2)
S_DB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_DB, sr=fs, hop_length=512, x_axis='time', y_axis='mel', ax=ax3, fmax=fs/2)
ax3.set_title('Mel Spectrogram')
fig.colorbar(img, ax=ax3, format='%+2.0f dB')

# 첫 번째 서브플롯의 위치와 크기 조정
# 첫 번째 서브플롯을 두 번째 서브플롯과 동일한 너비로 만들기 위해
# ax2.get_position()을 호출하여 두 번째 서브플롯의 위치 정보를 얻습니다.
pos_ax2 = ax2.get_position()
ax1.set_position([pos_ax2.x0, pos_ax2.y0 + pos_ax2.height + 0.1, pos_ax2.width, pos_ax2.height])

# ax3 (세 번째 서브플롯)의 위치를 조정합니다. 이는 ax2와 동일한 높이에서, 하지만 아래에 위치하도록 조정합니다.
ax3.set_position([pos_ax2.x0, pos_ax2.y0 - pos_ax2.height - 0.1, pos_ax2.width, pos_ax2.height])

plt.show()
