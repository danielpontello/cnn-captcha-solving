import plaidml.keras
plaidml.keras.install_backend()

import cv2
import glob
import os
import random
import string
import logging
import time

import numpy as np
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from network_model import NetworkModel

# CONSTANTES ============================================
# fixamos a seed para manter os resultados reproduzíveis
random.seed(71)

# caracteres permitidos no CAPTCHA
allowed_chars = string.ascii_lowercase + string.digits

def encode(char):
    '''Converte um caractere para uma representação one-hot'''
    arr = np.zeros((len(allowed_chars),), dtype="uint8")
    index = allowed_chars.index(char)
    arr[index] = 1
    return arr

def load_dataset(num):
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

        for file in files[:num]:
            image = cv2.imread(path + file)
            resized = cv2.resize(image, (30, 30))

            label = encode(char)

            data.append(resized)
            labels.append(label)
    return data, labels

def normalize_samples(data, labels):
    '''Normaliza as imagens, de um intervalo [0-255] para [0-1]
    e converte os dados para o formato NumPy'''
    n_data = np.array(data, dtype="float") / 255.0
    n_labels = np.array(labels)
    return n_data, n_labels

def train_network(train_x, train_y, validation_x, validation_y, epochs, learning_rate, batch_size, model_name):
    '''Treina a rede neural, salvando o histórico num .csv'''
    
    # caminho para salvar os modelos e os resultados
    mod_path = "../models/" + model_name + "/"

    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)

    # otimização: descida de gradiente estocástica
    sgd = SGD(lr=learning_rate)

    # compila o modelo da rede neural (definido em network_model)
    model = NetworkModel.build(30, 30, 3, len(allowed_chars))
    model.compile(loss='categorical_crossentropy', 
                optimizer=sgd, 
                metrics=['accuracy'])

    # callback para salvamento dos dados em .csv
    csv_logger = CSVLogger(mod_path + f'results.csv', separator=";")

    # callback para salvar sempre o melhor modelo
    checkpoint = ModelCheckpoint(filepath=mod_path + f'model.hdf5', 
                monitor='val_acc', 
                verbose=1, 
                save_best_only=True)

    # treinamento
    model.fit(train_x, train_y, 
                validation_data=(validation_x, validation_y), 
                batch_size=batch_size, 
                epochs=epochs, 
                verbose=1,
                callbacks=[checkpoint, csv_logger])

if __name__ == "__main__":
    num_samples = 2000
    epochs = 2
    learning_rate = 1e-3
    batch_size = 128
    validation_split=0.66

    print("Carregando dataset...")
    data, labels = load_dataset(num_samples)

    print("Normalizando amostras...")
    n_data, n_labels = normalize_samples(data, labels)

    print("Separando em treinamento e validação...")
    (train_x, validation_x, train_y, validation_y) = train_test_split(n_data, n_labels, test_size=0.3, random_state=42)

    print("Treinando rede...")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"model-{timestr}"
    train_network(train_x, train_y, validation_x, validation_y, epochs, learning_rate, batch_size, model_name)

    print("Treinamento concluído!")
    print(f"Os resultados foram salvos na pasta 'models/{model_name}'")
