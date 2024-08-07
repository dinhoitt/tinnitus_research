import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from pydub import AudioSegment
import scipy.signal as signal
from pydub.generators import Sine, WhiteNoise

# 오디오 파일 읽기 함수
def read_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    fs, data = wavfile.read(file_path)
    return audio, fs, data

# 스테레오 데이터를 각 채널로 분리하는 함수
def separate_stereo_channels(data):
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    return left_channel, right_channel

# 시간 축 생성 함수
def create_time_axis(data, fs):
    return np.arange(0, len(data)) / fs

# 노치 필터의 옥타브 폭 설정 함수
def get_octave_width(notch_freq):
    octave_ratio = 2 ** (1/12)  # 반음 간격의 비율
    options = {
        1: 12,
        2: 6,
        3: 3
    }
    print("노치 필터의 폭을 선택하세요:")
    print("1: 1옥타브")
    print("2: 1/2옥타브")
    print("3: 1/4옥타브")
    
    while True:
        try:
            choice = int(input("선택 (1, 2, 또는 3): "))
            if choice in options:
                return notch_freq * (octave_ratio ** options[choice] - octave_ratio ** (-options[choice]))
            else:
                print("유효하지 않은 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력해야 합니다. 다시 시도하세요.")

# 노치 필터 적용 함수
def apply_notch_filter(data, notch_freq, notch_width, fs):
    quality_factor = notch_freq / notch_width
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    return signal.filtfilt(b, a, data)

# 핑크 노이즈 생성 함수
def generate_pink_noise(duration_ms, fs):
    samples = int(duration_ms * fs / 1000.0)
    uneven = samples % 2
    X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # Power spectrum density
    y = (ifft(X / S)).real
    if uneven:
        y = y[:-1]
    return np.int16(y * 32767 / np.max(np.abs(y)))

# 밴드패스 필터 적용 함수
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# 구간별 평균 볼륨 계산 함수
def calculate_segment_volume(audio, segment_duration_ms):
    segments = []
    for i in range(0, len(audio), segment_duration_ms):
        segment = audio[i:i+segment_duration_ms]
        segments.append(segment.dBFS)
    return segments

# 필터링된 데이터를 저장하는 함수
def save_filtered_data(filtered_data, file_path, fs):
    wavfile.write(file_path, fs, filtered_data)

# 메인 코드
def main():
    file_path = "C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.wav"
    audio, fs, data = read_audio_file(file_path)
    
    left_channel, right_channel = separate_stereo_channels(data)
    
    # 필터링을 위한 사용자 입력
    notch_freq = int(input("이명 주파수를 입력하세요(예: 880): "))
    notch_width = get_octave_width(notch_freq)
    ear_choice = int(input("이명이 들리는 귀를 선택하세요 (왼쪽: 1, 오른쪽: 2): "))
    sound_choice = int(input("사용할 소리를 선택하세요 (바이노럴 비트: 1, 화이트 노이즈: 2, 핑크 노이즈: 3): "))
    
    if sound_choice == 1:
        wave_type = int(input("원하는 웨이브 타입을 선택하세요 (알파: 1, 베타: 2, 감마: 3): "))
        wave_freq_diff = {1: 10, 2: 20, 3: 40}.get(wave_type, 10)
    
    filtered_data_left = apply_notch_filter(left_channel, notch_freq, notch_width, fs)
    filtered_data_right = apply_notch_filter(right_channel, notch_freq, notch_width, fs)
    
    # 데이터를 int16 형식으로 변환
    filtered_data_left = np.int16(filtered_data_left / np.max(np.abs(filtered_data_left)) * 32767)
    filtered_data_right = np.int16(filtered_data_right / np.max(np.abs(filtered_data_right)) * 32767)
    
    filtered_data = np.column_stack((filtered_data_left, filtered_data_right))
    
    save_filtered_data(filtered_data, "C:\\Users\\taeso\\OneDrive\\바탕 화면\\filterd.wav", fs)
    
    # 필터링된 오디오 불러오기
    filtered_audio = AudioSegment.from_file("C:\\Users\\taeso\\OneDrive\\바탕 화면\\filterd.wav")
    
    # 구간별 평균 볼륨 계산
    segment_duration_ms = 10000  # 10초 단위로 구간 나누기
    segment_volumes = calculate_segment_volume(filtered_audio, segment_duration_ms)
    
    # 기존 오디오의 평균 볼륨 측정
    average_dbfs = filtered_audio.dBFS
    
    # 밴드패스 필터 설정
    lowcut = notch_freq - notch_width / 2
    highcut = notch_freq + notch_width / 2
    
    combined_audio = AudioSegment.silent(duration=len(filtered_audio))  # 결과 오디오 초기화
    
    if sound_choice == 1:
        # 이명 주파수에 따른 바이노럴 비트 주파수 계산
        frequency_left, frequency_right = (notch_freq, notch_freq + wave_freq_diff) if ear_choice == 1 else (notch_freq + wave_freq_diff, notch_freq)
    
        # 각 구간의 평균 볼륨을 기반으로 사인파 합성
        for i, segment_volume in enumerate(segment_volumes):
            # 현재 구간의 길이를 계산 (구간의 길이가 남은 오디오 길이를 초과하지 않도록 함)
            duration_ms = min(segment_duration_ms, len(filtered_audio) - i * segment_duration_ms)
            
            # 현재 구간의 시작 시간(ms)
            start_ms = i * segment_duration_ms
            
            # 현재 구간의 종료 시간(ms)
            end_ms = start_ms + duration_ms
            
            # 오른쪽 채널의 사인파 생성 및 볼륨 조절 (구간 볼륨 - 5dB)
            sine_right = Sine(frequency_right).to_audio_segment(duration=duration_ms).apply_gain(segment_volume - 5)
            
            # 왼쪽 채널의 사인파 생성 및 볼륨 조절 (구간 볼륨 - 5dB)
            sine_left = Sine(frequency_left).to_audio_segment(duration=duration_ms).apply_gain(segment_volume - 5)
            
            # 왼쪽과 오른쪽 사인파를 결합하여 스테레오 사인파 생성
            stereo_sine = AudioSegment.from_mono_audiosegments(sine_left, sine_right)
            
            # 기존 오디오의 현재 구간과 사인파를 합성
            combined_audio = combined_audio.overlay(filtered_audio[start_ms:end_ms], position=start_ms)
            
            # 스테레오 사인파를 기존 오디오에 합성
            combined_audio = combined_audio.overlay(stereo_sine, position=start_ms)
        
    elif sound_choice == 2:
        white_noise = WhiteNoise().to_audio_segment(duration=len(filtered_audio)).apply_gain(average_dbfs - 10)
        white_noise_data = np.array(white_noise.get_array_of_samples(), dtype=np.float32)
        bandpass_white_noise = bandpass_filter(white_noise_data, lowcut, highcut, fs)
        bandpass_white_noise_audio = AudioSegment(bandpass_white_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=white_noise.sample_width, channels=1)
        combined_audio = filtered_audio.overlay(bandpass_white_noise_audio)
    
    elif sound_choice == 3:
        duration_ms = len(filtered_audio)
        pink_noise_data = generate_pink_noise(duration_ms, fs).astype(np.float32)
        bandpass_pink_noise = bandpass_filter(pink_noise_data, lowcut, highcut, fs)
        pink_noise = AudioSegment(bandpass_pink_noise.astype(np.int16).tobytes(), frame_rate=fs, sample_width=2, channels=1)
        pink_noise = pink_noise.apply_gain(average_dbfs - 10)
        combined_audio = filtered_audio.overlay(pink_noise)
    
    else:
        print("유효하지 않은 선택입니다.")
        combined_audio = filtered_audio
    
    combined_audio.export("C:\\Users\\taeso\\OneDrive\\바탕 화면\\test_complete2354.mp3", format="mp3")

if __name__ == "__main__":
    main()
