import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras import backend as K

class NetworkModel:
    @staticmethod
    def build(width, height, depth, numchars, possiblechars):
        # cria o modelo
        model = Sequential()
        inputShape = (height, width, depth)

		# verifica o formato da imagem
        # (dependente do backend)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Primeiro bloco 
        model.add(Conv2D(16, (3, 3), input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.1))

        # Segundo bloco
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.1))

        # Terceiro Bloco
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.1))

        # Camada conectada
        model.add(Flatten())
        model.add(Dense(64*numchars))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(32*numchars))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(numchars*possiblechars))
        model.add(Activation('softmax'))
        
        return model