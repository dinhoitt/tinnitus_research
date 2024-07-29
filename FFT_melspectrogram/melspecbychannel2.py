import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

# mp3 파일 로딩
filename = "C:\\Users\\taeso\\OneDrive\\바탕 화면\\test_complete224.mp3"
data, fs = librosa.load(filename, sr=None, mono=False)

# 왼쪽 채널과 오른쪽 채널 분리
if data.ndim == 1:  # 모노 오디오 파일 처리
    left_channel = data
    right_channel = data
else:
    left_channel = data[0, :]
    right_channel = data[1, :]

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

# 첫 번째 서브플롯: 왼쪽 채널 스펙트로그램
ax1 = fig.add_subplot(gs[0])
D_left = librosa.stft(left_channel)
S_DB_left = librosa.amplitude_to_db(np.abs(D_left), ref=np.max)
img1 = librosa.display.specshow(S_DB_left, sr=fs, hop_length=512, x_axis='time', y_axis='log', ax=ax1)
ax1.set_title('Left Channel Spectrogram')
fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

# 두 번째 서브플롯: 오른쪽 채널 스펙트로그램
ax2 = fig.add_subplot(gs[1])
D_right = librosa.stft(right_channel)
S_DB_right = librosa.amplitude_to_db(np.abs(D_right), ref=np.max)
img2 = librosa.display.specshow(S_DB_right, sr=fs, hop_length=512, x_axis='time', y_axis='log', ax=ax2)
ax2.set_title('Right Channel Spectrogram')
fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

plt.tight_layout()
plt.show()
