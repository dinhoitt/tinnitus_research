"""
필터링 주파수 설정이 가능한 코드
"""
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.io import wavfile
from scipy.fft import fft, ifft
from pydub import AudioSegment
from scipy.signal import spectrogram

# wav 파일 로딩
fs, data = wavfile.read(r"C:\Users\박태수\Desktop\testfft.wav")

# 데이터를 Float 타입으로 변환
data_float = data.astype(np.float64)

# 데이터의 최대 절대값 찾기
max_val = np.max(np.abs(data_float))

# 신호를 최대 절대값으로 나누어 정규화
norm_data = data_float / max_val

# 시간 축을 생성
time = np.arange(0, len(data)) / fs

# 전체 플롯을 위한 Figure 생성
fig = plt.figure(figsize=(12, 8))

# 첫 번째 서브플롯: 시간 도메인 신호
ax1 = fig.add_subplot(211)
ax1.plot(time, norm_data)
ax1.set_title('Normalized Signal in Time Domain')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

# 두 번째 서브플롯: 스펙트로그램
ax2 = fig.add_subplot(212)
frequencies, times, Sxx = spectrogram(data, fs, nperseg=4096)
img = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno', norm=Normalize(vmin=-120, vmax=120))
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec]')
ax2.set_title('Spectrogram')

# 컬러바를 추가합니다.
cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
cbar.ax.set_ylabel('Intensity [dB]')

# 첫 번째 서브플롯의 위치를 조정합니다. (left, bottom, width, height)
ax1.set_position([0.125, 0.53, 0.6, 0.35])

# 두 번째 서브플롯의 위치를 첫 번째 서브플롯과 동일한 너비로 설정합니다.
ax2.set_position([0.125, 0.1, 0.6, 0.35])

plt.show()

# Hamming window 적용
hamming_window = np.hamming(len(norm_data))
windowed_data = norm_data * hamming_window

# fft 수행
fft_data = fft(windowed_data)

# 주파수 배열 생성
frequencies = np.linspace(0, fs, len(fft_data)//2)

# 사용자 입력을 검증하는 함수
def get_valid_input(prompt, valid_choices):
    while True:
        try:
            user_input = int(input(prompt))
            if user_input in valid_choices:
                return user_input
            else:
                print("유효하지 않은 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력해야 합니다. 다시 시도하세요.")

# 노치필터 옥타브 설정 함수
def get_octave_width(notch_freq):
    octave_ratio = 2 ** (1/12)  # 반음 간격의 비율
    print("노치 필터의 폭을 선택하세요:")
    print("1: 1옥타브")
    print("2: 1/2옥타브")
    print("3: 1/4옥타브")
    while True:
        try:
            choice = int(input("선택 (1, 2, 또는 3): "))
            if choice == 1:
                return notch_freq * (octave_ratio**12 - octave_ratio**(-12))
            elif choice == 2:
                return notch_freq * (octave_ratio**6 - octave_ratio**(-6))
            elif choice == 3:
                return notch_freq * (octave_ratio**3 - octave_ratio**(-3))
            else:
                print("유효하지 않은 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력해야 합니다. 다시 시도하세요.")


def processFrequencyBand(fft_data, frequencies, low, high, factor):
    # 주파수 범위에 해당하는 인덱스 찾기
    idx_band = np.where((frequencies >= low) & (frequencies <= high))

    # 해당 주파수 범위의 FFT 계수에 factor를 곱함
    fft_data[idx_band] *= factor

# FFT 데이터 준비
fft_data_half = fft_data[:len(fft_data)//2]

# 주파수 배열 생성
frequencies = np.linspace(0, fs, len(fft_data_half))

# processFrequencyBand 함수 적용
processFrequencyBand(fft_data_half, frequencies, 1500, 2500, 5)

# 노치 필터 적용
notch_freq = int(input("노치 필터링할 중심 주파수를 입력하세요 (예: 880): "))
notch_width = get_octave_width(notch_freq)

notch_lower = notch_freq - (notch_width / 2)
notch_upper = notch_freq + (notch_width / 2)
notch_filter = (frequencies > notch_lower) & (frequencies < notch_upper)
fft_data_filtered_half = fft_data[:len(fft_data)//2]
fft_data_filtered_half[notch_filter] = 0

# 대칭성을 유지하면서 전체 FFT 데이터 길이로 필터링된 데이터를 확장
fft_data_filtered = np.concatenate((fft_data_filtered_half, np.conj(fft_data_filtered_half[-2:0:-1])))

# 필터링된 신호를 시간 도메인으로 변환
filtered_data = ifft(fft_data_filtered)

# 실수 부분만 사용하여 시간 도메인 신호 획득
filtered_data_real = np.real(filtered_data)

# 필터링된 신호의 길이에 맞춰 time 배열의 길이 조정
time = time[:len(filtered_data_real)]

# 전체 플롯을 위한 Figure 생성
fig = plt.figure(figsize=(12, 8))

# 첫 번째 서브플롯: 시간 도메인 신호
ax1 = fig.add_subplot(211)
ax1.plot(time, filtered_data_real)
ax1.set_title('Filtered Signal in Time Domain')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

# 두 번째 서브플롯: 스펙트로그램
ax2 = fig.add_subplot(212)
frequencies, times, Sxx = spectrogram(filtered_data_real, fs, nperseg=4096)
img = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno', norm=Normalize(vmin=-300, vmax=120))
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec]')
ax2.set_title('Spectrogram')

# 컬러바를 추가합니다.
cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
cbar.ax.set_ylabel('Intensity [dB]')

# 첫 번째 서브플롯의 위치를 조정합니다. (left, bottom, width, height)
ax1.set_position([0.125, 0.53, 0.6, 0.35])

# 두 번째 서브플롯의 위치를 첫 번째 서브플롯과 동일한 너비로 설정합니다.
ax2.set_position([0.125, 0.1, 0.6, 0.35])

plt.show()

# 정규화 해제
filtered_data_unnormalized = filtered_data_real * max_val

# 16비트 정수형으로 변환
filtered_data_int16 = np.int16(filtered_data_unnormalized)

# 필터링된 신호를 WAV 파일로 저장
wavfile.write(r"C:\Users\박태수\Desktop\testfft_filtered_with_equalizing.wav", fs, filtered_data_int16)


