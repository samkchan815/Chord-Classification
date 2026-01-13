import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

shape = (1025, 97)
batchSize = 32

imageGenerator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_dataset = imageGenerator.flow_from_directory(
    directory="Audio_Files/Spectograms",
    batch_size=batchSize,
    target_size=shape,
    subset = "training",
    color_mode="grayscale",
    class_mode="binary")

validation_dataset = imageGenerator.flow_from_directory(
    directory="Audio_Files/Spectograms",
    batch_size=batchSize,
    target_size=shape,
    subset = "validation",
    color_mode="grayscale",
    class_mode="binary")

batch1 = train_dataset[0]

img = batch1[0][5] # first img in batch
lab = batch1[1][5] # label of first img in batch

print(img.shape)
plt.imshow(img)
plt.title(lab)
plt.axis('off')
plt.show()

# model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model = keras.models.Sequential([
    keras.layers.Input(shape=(1025, 97, 1)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(256, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(512, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss = "BinaryCrossentropy", 
              optimizer=optimizer, 
              metrics=["accuracy", keras.metrics.AUC(name = "auc")]
              )

model.summary()

os.makedirs("models", exist_ok=True)
best_model_file = "models/best_cnn_model.h5"
best_model = ModelCheckpoint(
    best_model_file,
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    callbacks=[best_model]
          )

