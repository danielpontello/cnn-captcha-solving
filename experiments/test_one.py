import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import argparse
import imutils
import os
import cv2
import sys
import string
import timeit
import random

from os import listdir
from os.path import splitext
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from difflib import SequenceMatcher
from memory_profiler import profile

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

def load_sample():
    '''Carrega os dados e os rótulos de cada letra em dois vetores,
    data e labels. O parâmetro num limita quantas amostras de cada
    letra devem ser carregadas, para reduzir o uso de memória.'''
    # caminho dos dados
    seg_path = "../dataset/segmented/"

    char = random.choice(allowed_chars)

    path = seg_path + char + "/"
    files = os.listdir(path)

    file = random.choice(files)

    image = cv2.imread(path + file)
    resized = cv2.resize(image, (30, 30))

    return resized, char

@profile
def predict(model, sample):
    sample = sample.astype("float32") / 255.0
    sample = np.expand_dims(sample, axis=0)

    out = model.predict(sample)
    decoded = decode(out)
    return decoded

if __name__ == "__main__":
    min_delta_vals = ["1e-05", "1e-06", "1e-07"]
    patience_vals = ["10"]

    
    print(f"Carregando modelo model-md[1e05]-pt[5]...")
    model = load_model(f'../models/model-md[1e-05]-pt[5]/model.hdf5')

    print("Carregando dataset...")

    sample, label = load_sample()    
    decoded = predict(model, sample)