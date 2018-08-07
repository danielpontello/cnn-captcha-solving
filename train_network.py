import cv2
import glob
import random

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from network_model import NetworkModel

# fixamos a seed para manter os resultados reproduzíveis
random.seed(1337)

data = []
labels = []

filenames = glob.glob("captchas/*.png")

def encode(number):
    numlist = list(str(number))

    encoded = []
    for num in numlist:
        arr = np.zeros((10,), dtype=int)
        arr[int(num)] = 1
        encoded.extend(arr)
    return encoded

print("carregando labels...")
with open("captchas/labels.txt", "r") as label_file:
    raw_labels = label_file.read().split("\n")

    # converte cada captcha para uma representação binária
    for label in raw_labels:
        enc_label = encode(label)
        labels.append(enc_label)

print("carregando imagens")
for file in filenames:
    image = cv2.imread(file)
    data.append(image)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print(data.shape)

print("carregando modelo")
model = NetworkModel.build(140, 80, 3, 4, 10)

model.fit(data, labels, batch_size=400, epochs=20, validation_split=0.1)
model.save('model.mdl')