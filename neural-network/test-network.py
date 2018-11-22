# Descomente as linhas abaixo
# aceleração de hardware em GPUs AMD
# import plaidml.keras
# plaidml.keras.install_backend()

import numpy as np
import argparse
import imutils
import os
import cv2
import sys
import string

from os import listdir
from os.path import splitext
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
# caminho do dataset
raw_path = "../dataset/raw/"
seg_path = "../dataset/segmented/"
tst_path = "../dataset/test/"
res_path = "../results/"

allowed_chars = string.ascii_lowercase + string.digits

def decode(array):
    index = np.argmax(array)
    return allowed_chars[index]

def load_dataset(start, end):
    '''Carrega os dados e os rótulos de cada letra em dois vetores,
    data e labels. O parâmetro num limita quantas amostras de cada
    letra devem ser carregadas, para reduzir o uso de memória.'''
    # caminho dos dados
    seg_path = "../dataset/segmented/"

    data = []
    labels = []

    for char in allowed_chars:
        print(f"Carregando dados do caractere '{char}'")

        path = seg_path + char + "/"
        files = os.listdir(path)

        for file in files[start:end]:
            image = cv2.imread(path + file)
            resized = cv2.resize(image, (30, 30))

            label = char

            data.append(resized)
            labels.append(label)
    return data, labels

if __name__ == "__main__":
    print("Carregando modelo...")
    model = load_model('../models/model-20181026-133514/model.hdf5')

    print("Carregando dataset...")
    data, labels = load_dataset(2000, 3320)

    y_true = []
    y_pred = []

    right = 0
    total = 0

    print("Efetuando testes...")
    for i in range(len(data)):
        sample = data[i]
        label = labels[i]

        sample = sample.astype("float32") / 255.0
        sample = np.expand_dims(sample, axis=0)

        out = model.predict(sample)
        decoded = decode(out)

        if label == decoded:
            right += 1
        total += 1

        y_true += label
        y_pred += decoded

    print("Calculando Matriz de Confusão...")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(allowed_chars))
    norm_conf_matrix = normalize(conf_matrix, norm='l1')
    np.savetxt("../results/confusion_matrix_artificial.csv", conf_matrix, delimiter = ";", fmt = "%.4f")
    np.savetxt("../results/confusion_matrix_artificial_norm.csv", norm_conf_matrix, delimiter = ";", fmt = "%.4f")

    print(f"Taxa de acerto: {right/total}%")