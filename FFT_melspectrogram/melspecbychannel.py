import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
import librosa
import librosa.display

# wav 파일 로딩
fs, data = wavfile.read("C:\\Users\\taeso\\OneDrive\\바탕 화면\\filtered.wav")

# 왼쪽 채널과 오른쪽 채널 분리
left_channel = data[:, 0]
right_channel = data[:, 1]

# 데이터를 Float 타입으로 변환 및 정규화
left_channel = left_channel.astype(np.float64)
right_channel = right_channel.astype(np.float64)

# 데이터의 최대 절대값 찾기
max_val_left = np.max(np.abs(left_channel))
max_val_right = np.max(np.abs(right_channel))

# 신호를 최대 절대값으로 나누어 정규화
left_channel = left_channel / max_val_left
right_channel = right_channel / max_val_right

# 전체 플롯을 위한 Figure 생성
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])  # 2개의 행과 1개의 열

# 첫 번째 서브플롯: 왼쪽 채널 멜 스펙트로그램
ax1 = fig.add_subplot(gs[0])
S_left = librosa.feature.melspectrogram(y=left_channel, sr=fs, n_mels=128, fmax=fs/2)
S_DB_left = librosa.power_to_db(S_left, ref=np.max)
img1 = librosa.display.specshow(S_DB_left, sr=fs, hop_length=512, x_axis='time', y_axis='mel', ax=ax1, fmax=fs/2)
ax1.set_title('Left Channel Mel Spectrogram')
fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

# 두 번째 서브플롯: 오른쪽 채널 멜 스펙트로그램
ax2 = fig.add_subplot(gs[1])
S_right = librosa.feature.melspectrogram(y=right_channel, sr=fs, n_mels=128, fmax=fs/2)
S_DB_right = librosa.power_to_db(S_right, ref=np.max)
img2 = librosa.display.specshow(S_DB_right, sr=fs, hop_length=512, x_axis='time', y_axis='mel', ax=ax2, fmax=fs/2)
ax2.set_title('Right Channel Mel Spectrogram')
fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

plt.tight_layout()
plt.show()
