import librosa
import numpy as np
import os
from glob import glob
import cv2

import tensorflow
from tensorflow import keras
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

model = tensorflow.keras.models.load_model("best_cnn_model.h5")
shape = (1025, 97)

def prepare_audio(pathForAudio):
  y, sr = librosa.load(pathForAudio)
  D = librosa.stft(y)
  s_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)
  imageAudio = (s_db*255).astype(np.uint8)

  resizedImage = cv2.resize(imageAudio, shape, interpolation = cv2.INTER_AREA)

  imgResult = img_to_array(resizedImage)
  imgResult = np.expand_dims(imgResult, axis=0)
  imgResult = imgResult/255.

  return imgResult

def prepare_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (shape[1], shape[0]))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # channel
    img = np.expand_dims(img, axis=0)   # batch
    return img


dataset = "Audio_Files/chords_testing_dataset"
classes = ["Major", "Minor"]

results = []

for label_name in classes:
    label_dir = os.path.join(dataset, label_name)
    image_files = glob(os.path.join(label_dir, "*.png"))

    true_label = 0 if label_name == "Major" else 1

    for audio_file in glob(os.path.join(label_dir, "*.wav")):

        img = prepare_audio(audio_file)
        pred = model.predict(img, verbose=0)[0][0]

        pred_label = 1 if pred >= 0.5 else 0
        correct = (pred_label == true_label)

        results.append({
            "file": os.path.basename(audio_file),
            "true_label": label_name,
            "predicted_label": "Minor" if pred_label == 1 else "Major",
            "confidence": float(pred),
            "correct": correct
        })

df = pd.DataFrame(results)
df.head()

accuracy = df["correct"].mean()
print(f"Test Accuracy: {accuracy:.4f}")


from sklearn.metrics import confusion_matrix, classification_report

y_true = df["true_label"].map({"Major": 0, "Minor": 1})
y_pred = df["predicted_label"].map({"Major": 0, "Minor": 1})

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Major", "Minor"]))


