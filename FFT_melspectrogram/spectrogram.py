# -*- coding: utf-8 -*-
"""
Life is what you make of it!

Written by @dinho_itt(ig_id)
"""
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from pydub import AudioSegment

# mp3 file 로딩
audio = AudioSegment.from_file(r"C:\Users\Home\Desktop\audio_data\47904428 박다인 비음 심각 보상조음 glottal stop 발달적 조음오류.MP3")
# mp3 파일을 모노로 변환하고 wav형식으로 변환
audio = audio.set_channels(1)
audio.export(r"C:\Users\Home\Desktop\audio_data\47904428 박다인 비음 심각 보상조음 glottal stop 발달적 조음오류.wav", format="wav")

# wav 파일 로딩
fs, data = wavfile.read(r"C:\Users\Home\Desktop\audio_data\47904428 박다인 비음 심각 보상조음 glottal stop 발달적 조음오류.wav")

# 시간 축을 생성합니다.
time = np.arange(0, len(data)) / fs

# 스펙토그램 생성
plt.figure(figsize=(24,8))
frequencies, times, Sxx = spectrogram(data, fs, nperseg=128)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='plasma', norm=Normalize(vmin=-60, vmax=60))
plt.title('Spectrogram')
plt.ylabel('Frequency[Hz]')
plt.xlabel('Time[sec]')
plt.colorbar(format='%+2.0f dB')
plt.show()