import torch
import torchaudio
import torchvision
import pyaudio
import wave
FORMAT = pyaudio.paInt16
file = "umang.wav"
duration = 10
freq = 44100
p = pyaudio.PyAudio()
channels = 2
sample_rate = 44100
chunk = 1024
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
