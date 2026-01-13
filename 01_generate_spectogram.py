import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import librosa

# Generate spectograms for all major chords
path_to_major_spectograms = "Audio_Files/Major_Spectograms/"
isExist = os.path.exists(path_to_major_spectograms)
if not isExist:
  os.makedirs(path_to_major_spectograms)
  print("The new directory is created!")

major_audio_files = glob("Audio_Files/Major/*.wav")

for file in major_audio_files:
  y, sr = librosa.load(file)
  D = librosa.stft(y)
  S_bd = librosa.amplitude_to_db(np.abs(D), ref=np.max)
  image_audio = (S_bd *255).astype(np.uint8)

  filename = os.path.basename(file)
  out_path = os.path.join(
        path_to_major_spectograms,
        filename + ".png"
    )

  cv2.imwrite(out_path, image_audio)
  print(filename)

# Generate spectograms for all minor chords
path_to_minor_spectograms = "Audio_Files/Minor_Spectograms/"
isExist = os.path.exists(path_to_minor_spectograms)
if not isExist:
  os.makedirs(path_to_minor_spectograms)
  print("The new directory is created!")

minor_audio_files = glob("Audio_Files/Minor/*.wav")

for file in minor_audio_files:
  y, sr = librosa.load(file)
  D = librosa.stft(y)
  S_bd = librosa.amplitude_to_db(np.abs(D), ref=np.max)
  image_audio = (S_bd *255).astype(np.uint8)

  filename = os.path.basename(file)
  out_path = os.path.join(
        path_to_minor_spectograms,
        filename + ".png"
    )

  cv2.imwrite(out_path, image_audio)
  print(filename)



