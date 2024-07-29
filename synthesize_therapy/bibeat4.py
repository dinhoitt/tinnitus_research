import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from pydub import AudioSegment
import scipy.signal as signal
from pydub.generators import Sine, WhiteNoise

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

def apply_notch_filter(data, notch_freq, notch_width, fs):
    quality_factor = notch_freq / notch_width
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def generate_pink_noise(duration_ms, fs):
    samples = int(duration_ms * fs / 1000.0)
    uneven = samples % 2
    X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # Power spectrum density
    y = (ifft(X / S)).real
    if uneven:
        y = y[:-1]
    return np.int16(y * 32767 / np.max(np.abs(y)))

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

# 필터링
notch_freq = int(input("이명 주파수를 입력하세요(예: 880): "))
notch_width = get_octave_width(notch_freq)  # 한 번만 노치 필터의 폭 선택
ear_choice = int(input("이명이 들리는 귀를 선택하세요 (왼쪽: 1, 오른쪽: 2): "))
sound_choice = int(input("사용할 소리를 선택하세요 (바이노럴 비트: 1, 화이트 노이즈: 2, 핑크 노이즈: 3): "))

if sound_choice == 1:
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

filtered_data_left = apply_notch_filter(left_channel, notch_freq, notch_width, fs)
filtered_data_right = apply_notch_filter(right_channel, notch_freq, notch_width, fs)

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

# 밴드패스 필터 설정
lowcut = notch_freq - notch_width / 2
highcut = notch_freq + notch_width / 2

if sound_choice == 1:
    # 이명 주파수에 따른 바이노럴 비트 주파수 계산 및 생성
    if ear_choice == 1:
        frequency_left = notch_freq  # 이명이 들리는 왼쪽 귀
        frequency_right = notch_freq + wave_freq_diff  # 오른쪽 귀에 바이노럴 비트
    else:
        frequency_right = notch_freq  # 이명이 들리는 오른쪽 귀
        frequency_left = notch_freq + wave_freq_diff  # 왼쪽 귀에 바이노럴 비트

    duration_ms = len(filtered_audio)  # 기존 오디오와 동일한 길이

    # 사인파 생성 및 볼륨 조절
    # 기존 오디오 볼륨보다 10dB 낮게 설정
    sine_right = Sine(frequency_right).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 10)
    sine_left = Sine(frequency_left).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 10)

    # 스테레오 사인파 생성
    stereo_sine = AudioSegment.from_mono_audiosegments(sine_left, sine_right)

    # 기존 오디오와 사인파 합성
    combined_audio = filtered_audio.overlay(stereo_sine)

elif sound_choice == 2:
    # 화이트 노이즈 생성 및 볼륨 조절
    white_noise = WhiteNoise().to_audio_segment(duration=len(filtered_audio)).apply_gain(average_dbfs - 10)
    white_noise_data = np.array(white_noise.get_array_of_samples(), dtype=np.float32)
    bandpass_white_noise = bandpass_filter(white_noise_data, lowcut, highcut, fs)
    bandpass_white_noise_audio = AudioSegment(bandpass_white_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=white_noise.sample_width, channels=1)
    combined_audio = filtered_audio.overlay(bandpass_white_noise_audio)

elif sound_choice == 3:
    # 핑크 노이즈 생성 및 볼륨 조절
    duration_ms = len(filtered_audio)  # 기존 오디오와 동일한 길이
    pink_noise_data = generate_pink_noise(duration_ms, fs).astype(np.float32)
    bandpass_pink_noise = bandpass_filter(pink_noise_data, lowcut, highcut, fs)
    pink_noise = AudioSegment(bandpass_pink_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=2, channels=1)
    pink_noise = pink_noise.apply_gain(average_dbfs - 10)
    combined_audio = filtered_audio.overlay(pink_noise)

else:
    print("유효하지 않은 선택입니다.")
    combined_audio = filtered_audio

# 결과 파일로 저장
combined_audio.export("C:\\Users\\taeso\\OneDrive\\바탕 화면\\test_complete224.mp3", format="mp3")
