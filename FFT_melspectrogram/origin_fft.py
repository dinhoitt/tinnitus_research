import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from pydub import AudioSegment

# mp3 file 로딩
audio = AudioSegment.from_file(r"C:\Users\Home\Desktop\testfft.MP3")
# mp3 파일을 모노로 변환하고 wav형식으로 변환
""" 
기본적으로 FFT를 하려먼 1차원 음성 신호여야 합니다. 
MP3는 대부분 스테레오 타입이 많아서 좌우로 분리된 오디오 채널이 많습니다.
MP3는 압축 포맷이기 때문에 압축되지 않은 오디오 파일 형식인 WAVV파일을 이용해야 합니다.
"""
audio = audio.set_channels(1)
audio.export("C:/Users/Home/Desktop/testfft.wav", format="wav")

# wav 파일 로딩
fs, data = wavfile.read("C:/Users/Home/Desktop/testfft.wav")

# 데이터를 Float 타입으로 변환
data_float = data.astype(np.float64)

# 데이터의 최대 절대값 찾기
max_val = np.max(np.abs(data_float))

# 신호를 최대 절대값으로 나누어 정규화
norm_data = data_float / max_val

# 시간 축을 생성
time = np.arange(0, len(data)) / fs

#정규화된 시간 도메인 신호 그림 생성
plt.figure(figsize=(12, 6))
plt.plot(time, norm_data)
plt.title('Nomalized Signal in Time Domain')
plt.xlabel('Time(seconds)')
plt.ylabel('Amplitude')
plt.show()

# Hamming window 적용
hamming_window = np.hamming(len(norm_data))
windowed_data = norm_data * hamming_window

# fft 수행
fft_data = fft(windowed_data)

# 주파수 도메인 스펙트럼 plot
frequencies = np.linspace(0, fs, len(fft_data))
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_data)[:len(frequencies)//2]) # plot only the positive frequencies
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# 주파수 배열 생성
frequencies = np.linspace(0, fs, len(fft_data)//2)

# 밴드 패스 필터 적용
f_low = 1500
f_high = 6000
band_pass_filter = (frequencies >= f_low) & (frequencies <= f_high)
fft_data_filtered_half = fft_data[:len(fft_data)//2]
fft_data_filtered_half[(frequencies < f_low) | (frequencies > f_high)] = 0

# 대칭성을 유지하면서 전체 FFT 데이터 길이로 필터링된 데이터를 확장
fft_data_filtered = np.concatenate((fft_data_filtered_half, np.conj(fft_data_filtered_half[-2:0:-1])))

# 필터링된 신호를 시간 도메인으로 변환
filtered_data = ifft(fft_data_filtered)

# 실수 부분만 사용하여 시간 도메인 신호 획득
filtered_data_real = np.real(filtered_data)

# 주파수 도메인의 필터링된 신호 그림 생성
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_data)[:len(frequencies)//2])
plt.title('Filtered Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# 필터링된 신호의 길이에 맞춰 time 배열의 길이 조정
time = time[:len(filtered_data_real)]

#필터링된 신호를 시간 도메인에서 그림 생성
plt.figure(figsize=(12, 6))
plt.plot(time, filtered_data_real)
plt.title('Filtered Signal in Time Domain')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

# 정규화 해제
filtered_data_unnormalized = filtered_data_real * max_val

# 16비트 정수형으로 변환
filtered_data_int16 = np.int16(filtered_data_unnormalized)

# 필터링된 신호를 WAV 파일로 저장
wavfile.write('C:/Users/Home/Desktop/testfft_filtered.wav', fs, filtered_data_int16)



