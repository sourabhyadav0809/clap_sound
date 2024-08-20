import simpleaudio as sa
from scipy.io.wavfile import write
import sounddevice as sd
import wavio as wv
import torchaudio
from torchaudio import transforms
import PIL
import torch
import matplotlib.pyplot as plt 
import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import subprocess
# filename = "clap.wav"
from numpy.linalg import norm
def record_and_save_audio(recording_):

    freq = 44100
    
    # # Recording duration
    duration = 2
    
    # Start recorder with the given values 
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq), 
                       samplerate=freq, channels=1)
    print('start clapping')
    
    # Record audio for the given number of seconds
    sd.wait()
    
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write(recording_, freq, recording)
    
    # Convert the NumPy array to audio file
    wv.write(recording_, recording, freq, sampwidth=2)



def get_embeddings(recording, img_path):
    sample_rate = 44100
    waveform, sample_rate = torchaudio.load(recording, normalize=True)
    transform = transforms.MelSpectrogram(sample_rate, n_mels= 32)
    mel_specgram = transform(waveform) 
    print(mel_specgram)
    mel_specgram = torch.squeeze(mel_specgram, 0)

        


    print(mel_specgram.size())

    # orch.unsqueeze(mel_specgram,1)

    plt.imshow(mel_specgram)
    plt.savefig(img_path)
    def return_image_embedding(model,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        curr_df = pd.DataFrame(preds[0]).T
        return curr_df
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    
    a = return_image_embedding(model,img_path)
    return a

b = get_embeddings("recording4.wav", "4.png")
def main():
    rec = "recording6.wav"
    record_and_save_audio(recording_=rec)
    a = get_embeddings(rec, f"{rec[:-4]}.png")
    
    cosine = np.dot(np.squeeze(a),np.squeeze(b))/((norm(a))*norm((b)))
    print(cosine)
    return cosine
direction = True

while True:
   
    def brightness(direction):
        if(direction==True):
            subprocess.run('osascript increase_brightness.applescript',shell=True)
            direction= False
            return False
        else:
           
            subprocess.run('osascript decrease_brightness.applescript',shell=True)

            

                
            direction = True
            return True

    consi= main()
    if(consi>0.99):
        
        direction = brightness(direction)

        
        
    else:
        print("didnt listen to claps")

