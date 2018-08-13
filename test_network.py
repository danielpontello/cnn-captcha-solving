import numpy as np
import argparse
import imutils
import cv2
import sys

from keras.preprocessing.image import img_to_array
from keras.models import load_model

kernel = (5, 5)
level = 4

def decode(array):
    return np.argmax(array)

image = cv2.imread(sys.argv[1], 0)
ret, image = cv2.threshold(image, 211, 255, cv2.THRESH_BINARY_INV)
image = cv2.erode(image, kernel, iterations = level)
image = cv2.dilate(image, kernel, iterations = level - 2)
image = img_to_array(image)
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)
print(image.shape)

model = load_model('model.mdl')
res = model.predict(image)

print(decode(res[0][0:10]), decode(res[0][10:20]), decode(res[0][20:30]), decode(res[0][30:40]))