import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from pydub import AudioSegment
from pydub.generators import Sine

audio = AudioSegment.from_file("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.wav")

audio = audio.set_channels(2)
audio.export("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.mp3", format="mp3")

original_audio = AudioSegment.from_file("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testbibeat.mp3")

# 기존 오디오의 평균 볼륨 측정
average_dbfs = original_audio.dBFS

# 사인파 생성 설정
frequency_right = 428  # 오른쪽 채널 주파수
frequency_left = 438   # 왼쪽 채널 주파수
duration_ms = len(original_audio)  # 기존 오디오와 동일한 길이

# 사인파 생성 및 볼륨 조절
# 기존 오디오 볼륨보다 5dB 낮게 설정
sine_right = Sine(frequency_right).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 5)
sine_left = Sine(frequency_left).to_audio_segment(duration=duration_ms).apply_gain(average_dbfs - 5)


# 스테레오 사인파 생성
stereo_sine = AudioSegment.from_mono_audiosegments(sine_left, sine_right)

# 기존 오디오와 사인파 합성
combined_audio = original_audio.overlay(stereo_sine)

# 결과 파일로 저장
combined_audio.export("C:\\Users\\taeso\\OneDrive\\바탕 화면\\testcomplete.mp3", format="mp3")