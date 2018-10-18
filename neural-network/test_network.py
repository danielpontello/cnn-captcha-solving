import plaidml.keras
plaidml.keras.install_backend()


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
from difflib import SequenceMatcher

import matplotlib.pyplot as plt

# parâmetros do blur
kernel = (3, 3)
level = 2

# caminho do dataset
raw_path = "../dataset/raw/"
seg_path = "../dataset/segmented/"
tst_path = "../dataset/test/"
res_path = "../results/"

allowed_chars = string.ascii_lowercase + string.digits

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def decode(array):
    index = np.argmax(array)
    return allowed_chars[index]

files = listdir(tst_path)

right = 0
total = 1

# log html
page_header = '''
 <html>
 <body style='font-family: Arial'>
 <table border='1' align='center'>
  <tr>
    <th>CAPTCHA</th>
    <th>Esperado</th>
    <th>Inferido</th>
    <th>Taxa de Acerto</th>
  </tr>
'''

page_footer = '''
</table>
</body>
</html>
'''

print("Carregando modelo...")
model = load_model('model.mdl')

data = []

for file in files:
    fullpath = tst_path + file
    image = cv2.imread(fullpath, 0)
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
        
        # plt.imshow(rgb)
        # plt.show()
        
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

    acc = similar(expected, final_str)

    data.append([fullpath, expected, final_str, acc])

    #print(f"Expected: {expected}")
    #print(f"Inferred: {final_str}")
    #print(f"Accuracy: {str(similar(expected, final_str))}")
    #print(f"Right: {right}/{total} ({str((right/total)*100)[:5]}%)")

with open(res_path + "results.html", "w") as html:
    html.write(page_header)

    for line in data:
        acc = line[3]

        if acc >= 0.75:
            tr = "<tr style='color:green;'>"
        elif acc >= 0.5:
            tr = "<tr style='color:orange;'>"
        else:
            tr = "<tr style='color:red;'>"
        html_str = tr + f"<td><img src='{'../' + line[0]}'></td><td style='font-family: monospace; font-size: 32px; color:black;'>{line[1]}</td><td style='font-family: monospace; font-size: 32px;'>{line[2][:4]}</td><td style='color:black;'>{str(acc*100)[:5]}%</td></tr>"
        html.write(html_str)
    html.write(page_footer)
