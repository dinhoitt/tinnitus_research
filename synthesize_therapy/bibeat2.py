import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from pydub import AudioSegment
import scipy.signal as signal
from pydub.generators import Sine

# 오디오 파일 읽기
audio = AudioSegment.from_file("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.wav")
fs, data = wavfile.read("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.wav")

# 스테레오 데이터를 각 채널로 분리
left_channel = data[:, 0]
right_channel = data[:, 1]

# 시간 축 생성
time = np.arange(0, len(data)) / fs

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

def apply_notch_filter(data, notch_freq, fs):
    notch_width = get_octave_width(notch_freq)
    quality_factor = notch_freq / notch_width
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# 필터링
notch_freq = int(input("이명 주파수를 입력하세요(예: 880): "))
ear_choice = int(input("이명이 들리는 귀를 선택하세요 (왼쪽: 1, 오른쪽: 2): "))
wave_type = int(input("원하는 웨이브 타입을 선택하세요 (알파: 1, 베타: 2, 감마: 3): "))

# 웨이브 타입에 따른 주파수 차이 설정
if wave_type == 1:
    wave_freq_diff = 10  # 알파 웨이브 (8-12Hz)
elif wave_type == 2:
    wave_freq_diff = 20  # 베타 웨이브 (12-30Hz)
elif wave_type == 3:
    wave_freq_diff = 40  # 감마 웨이브 (30-100Hz)
else:
    print("유효하지 않은 선택입니다. 기본값으로 알파 웨이브를 사용합니다.")
    wave_freq_diff = 10

filtered_data_left = apply_notch_filter(left_channel, notch_freq, fs)
filtered_data_right = apply_notch_filter(right_channel, notch_freq, fs)

# 데이터를 int16 형식으로 변환
filtered_data_left = np.int16(filtered_data_left / np.max(np.abs(filtered_data_left)) * 32767)
filtered_data_right = np.int16(filtered_data_right / np.max(np.abs(filtered_data_right)) * 32767)

# 스테레오 데이터로 결합
filtered_data = np.column_stack((filtered_data_left, filtered_data_right))

# 필터링된 데이터를 WAV 파일로 저장
wavfile.write("C:\\Users\\taeso\\OneDrive\\바탕 화면\\filterd.wav", fs, filtered_data)

# 필터링된 오디오 불러오기
filtered_audio = AudioSegment.from_file("C:\\Users\\taeso\\OneDrive\\바탕 화면\\filterd.wav")

# 기존 오디오의 평균 볼륨 측정
average_dbfs = filtered_audio.dBFS

# 이명 주파수에 따른 바이노럴 비트 주파수 계산 및 생성
if ear_choice == 1:
    frequency_left = notch_freq  # 이명이 들리는 왼쪽 귀
    frequency_right = notch_freq + wave_freq_diff  # 오른쪽 귀에 바이노럴 비트
else:
    frequency_right = notch_freq  # 이명이 들리는 오른쪽 귀
    frequency_left = notch_freq + wave_freq_diff  # 왼쪽 귀에 바이노럴 비트

duration_ms = len(filtered_audio)  # 기존 오디오와 동일한 길이

# 사인파 생성 및 볼륨 조절
# 기존 오디오 볼륨보다 5dB 낮게 설정
sine_right = Sine(frequency_right).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 5)
sine_left = Sine(frequency_left).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 5)

# 스테레오 사인파 생성
stereo_sine = AudioSegment.from_mono_audiosegments(sine_left, sine_right)

# 기존 오디오와 사인파 합성
combined_audio = filtered_audio.overlay(stereo_sine)

# 결과 파일로 저장
combined_audio.export("C:\\Users\\taeso\\OneDrive\\바탕 화면\\test_complete2.mp3", format="mp3")
