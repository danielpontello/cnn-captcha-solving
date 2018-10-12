import plaidml.keras
plaidml.keras.install_backend()

import cv2
import glob
import os
import random
import string
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

# caminho dos dados
seg_path = "../dataset/segmented/"

# épocas
EPOCHS = 64
# taxa de aprendizado
LR = 0.1
# decay
DECAY = 1e-6
# tamanho do batch
BS = 128

# parametros do filtro
kernel = (5, 5)
level = 4

# caracteres permitidos
allowed_chars = string.ascii_lowercase + string.digits

data = []
labels = []

def encode(char):
    arr = np.zeros((len(allowed_chars),), dtype="uint8")
    index = allowed_chars.index(char)
    arr[index] = 1
    return arr

print("Carregando dataset...")

for char in allowed_chars:
    print(f"Carregando dados do caractere '{char}'")

    path = seg_path + char + "/"
    files = os.listdir(path)

    for file in files:
        image = cv2.imread(path + file)
        resized = cv2.resize(image, (30, 30))

        label = encode(char)

        data.append(resized)
        labels.append(label)

print(f"{str(len(data))} amostras carregadas")

# as imagens são normalizadas para um range de [0:1]
print("Normalizando amostras...")

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print("Separando dados em treino e teste...")
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.25, random_state=42)

print("Carregando modelo...")
model = NetworkModel.build(30, 30, 3, len(allowed_chars))

# resumo do modelo
model.summary()

print("Treinando modelo...")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', 
                optimizer=sgd, 
                metrics=['accuracy'])
                
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BS, epochs=EPOCHS, verbose=1)

print("Salvando modelo resultante...")
model.save('model.mdl')
