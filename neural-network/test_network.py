import numpy as np
import argparse
import imutils
import cv2
import sys
import string

from os import listdir
from os.path import splitext
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import matplotlib.pyplot as plt

# parâmetros do blur
kernel = (3, 3)
level = 2

# caminho do dataset
raw_path = "../dataset/raw/"
seg_path = "../dataset/segmented/"
tst_path = "../dataset/test/"

allowed_chars = string.ascii_lowercase + string.digits

def decode(array):
    index = np.argmax(array)
    return allowed_chars[index]

files = listdir(tst_path)

right = 0
total = 1

print("Carregando modelo...")
model = load_model('model.mdl')

for file in files:
    image = cv2.imread(tst_path + file, 0)
    expected = splitext(file)[0]

    # blur
    k = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,k)

    # threshold
    ret, image = cv2.threshold(dst, 110, 255, cv2.THRESH_BINARY_INV)
    image = cv2.erode(image, kernel, iterations = level)

    connectivity = 4
    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    objects = []

    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]

        # remove pequenos ruídos
        if a > 50:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            objects.append((x, y, w, h))

    objects.sort(key=lambda t: t[0])

    letters = []

    for o in objects:
        x = o[0]
        y = o[1]
        w = o[2]
        h = o[3]

        img = image[y:y+h, x:x+w]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        rgb = cv2.resize(rgb, (30, 30))
        rgb = rgb.astype("float32") / 255.0
        rgb = np.expand_dims(rgb, axis=0)

        out = model.predict(rgb)
        decoded = decode(out)
        letters.append(decoded)

    final_str = ''.join(letters)

    if expected == final_str:
        right += 1
    total += 1

    print(f"Expected: {expected}")
    print(f"Inferred: {final_str}")
    print(f"Right: {right}/{total} ({right/total}%)")

'''


image = cv2.imread(sys.argv[1], 0)
#ret, image = cv2.threshold(image, 211, 255, cv2.THRESH_BINARY_INV)
#image = cv2.erode(image, kernel, iterations = level)
#image = cv2.dilate(image, kernel, iterations = level)
image = img_to_array(image)
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)
print(image.shape)

model = load_model('model.mdl')
res = model.predict(image)

def plot_res(res):        
    x = list(range(10))
    plt.subplot(221)
    plt.plot(x, res[0][0:10])
    plt.title("Dígito 1")
    plt.subplot(222)
    plt.plot(x, res[0][10:20])
    plt.title("Dígito 2")
    plt.subplot(223)
    plt.plot(x, res[0][20:30])
    plt.title("Dígito 3")
    plt.subplot(224)
    plt.plot(x, res[0][30:40])
    plt.title("Dígito 4")
    plt.show()

print(decode(res[0][0:10]), decode(res[0][10:20]), decode(res[0][20:30]), decode(res[0][30:40]))
plot_res(res)
'''