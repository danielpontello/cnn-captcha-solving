import plaidml.keras
plaidml.keras.install_backend()

import cv2
import glob
import os
import random
import coloredlogs
import logging

import numpy as np
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from network_model import NetworkModel
from time import time

coloredlogs.install(level='DEBUG')

# fixamos a seed para manter os resultados reproduzíveis
random.seed(71)

EPOCHS = 25
INIT_LR = 1e-3
BS = 16

kernel = (5, 5)
level = 4

data = []
labels = []

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

filenames = glob.glob("captchas/*.png")
filenames.sort(key=sortKeyFunc)
print(filenames)

def encode(number):
    numlist = list(str(number))

    encoded = []
    for num in numlist:
        arr = np.zeros((10,), dtype="uint8")
        arr[int(num)] = 1
        encoded.extend(arr)
    return encoded

print("carregando labels...")
with open("captchas/labels.txt", "r") as label_file:
    raw_labels = label_file.read().split("\n")

    # converte cada captcha para uma representação binária
    for label in raw_labels[:8000]:
        enc_label = encode(label)
        labels.append(enc_label)


print("carregando imagens...")
for file in filenames[:8000]:
    image = cv2.imread(file, 0)
    ret, image = cv2.threshold(image, 211, 255, cv2.THRESH_BINARY_INV)
    image = cv2.erode(image, kernel, iterations = level)
    image = cv2.dilate(image, kernel, iterations = level)
    image = img_to_array(image)
    data.append(image)

# as imagens são normalizadas para um range de [0:1]
print("normalizando imagens...")
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.25, random_state=42)

print("carregando modelo...")
model = NetworkModel.build(140, 80, 1, 4, 10)

# resumo do modelo
model.summary()

print("treinando modelo...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BS, epochs=EPOCHS, verbose=1)
model.save('model.mdl')