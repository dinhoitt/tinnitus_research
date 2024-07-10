from matplotlib.colors import Normalize
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# wav 파일 로딩 (pydub 사용)
audio = AudioSegment.from_file(r"C:\Users\Home\Desktop\audio_data\66569369 임지현 비음 중등-심각 발음 경도 ㅅㅈ_사탕.wav")
data = np.array(audio.get_array_of_samples())

# 스테레오인 경우 모노로 변환
if audio.channels == 2:
    data = data.reshape((-1, 2))
    data = data.mean(axis=1)

# 데이터 타입 변환 및 샘플 레이트 설정
data = data.astype(np.float32)
fs = audio.frame_rate

# nperseg 값 설정 (예: 0.1초의 샘플 길이)
nperseg = int(0.01 * fs)

# 스펙토그램 생성
frequencies, times, Sxx = spectrogram(data, fs, nperseg=nperseg)

# 스펙토그램 플롯팅
plt.figure(figsize=(24, 8))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno', norm=Normalize(vmin=-60, vmax=60))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f dB')
plt.show()