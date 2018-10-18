import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import random
import string

from PIL import Image
from os import listdir, makedirs
from collections import defaultdict
from os.path import join, isdir, splitext

# parâmetros do blur
kernel = (3, 3)
level = 2

# caminho do dataset
raw_path = "../dataset/raw/"
seg_path = "../dataset/segmented/"

allowed_chars = string.ascii_lowercase + string.digits

if not isdir(seg_path):
    makedirs(seg_path)

    for i in allowed_chars:
        makedirs(seg_path + "/" + i)

files = listdir(raw_path)

counts = defaultdict(int)

file = random.choice(files)
image = cv2.imread(raw_path + file, 0)

plt.subplot(5, 1, 1)
rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.imshow(rgb)

letters = splitext(file)[0]

# blur
k = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(image,-1,k)

plt.subplot(5, 1, 2)
rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
plt.imshow(rgb)

# threshold
ret, image = cv2.threshold(dst, 110, 255, cv2.THRESH_BINARY_INV)

plt.subplot(5, 1, 3)
rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.imshow(rgb)

image = cv2.erode(image, kernel, iterations = level)

plt.subplot(5, 1, 4)
rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.imshow(rgb)

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

num_detected = min(len(objects), 4)

rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

for i in range(num_detected):
    o = objects[i]
    x = o[0]
    y = o[1]
    w = o[2]
    h = o[3]

    cv2.rectangle(rgb,(x, y),(x+w, y+h),(255,0,0), 1)

plt.subplot(5, 1, 5)
plt.imshow(rgb)
plt.show()