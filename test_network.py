import numpy as np
import argparse
import imutils
import cv2
import sys

from keras.preprocessing.image import img_to_array
from keras.models import load_model

def decode(array):
    return np.argmax(array)

image = cv2.imread(sys.argv[1])
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)
print(image.shape)

model = load_model('model.mdl')
res = model.predict(image)

print(decode(res[0][0:10]))
print(decode(res[0][10:20]))
print(decode(res[0][20:30]))
print(decode(res[0][30:40]))