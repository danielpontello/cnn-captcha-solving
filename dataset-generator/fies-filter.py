import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
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

for file in files:
    image = cv2.imread(raw_path + file, 0)
    letters = splitext(file)[0]

    print(letters)

    # blur
    k = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,k)

    # threshold
    ret, image = cv2.threshold(dst, 110, 255, cv2.THRESH_BINARY_INV)
    image = cv2.erode(image, kernel, iterations = level)

    # plt.imshow(image)
    # plt.show()

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

    for i in range(num_detected):
        o = objects[i]
        x = o[0]
        y = o[1]
        w = o[2]
        h = o[3]

        img = image[y:y+h, x:x+w]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        letter = letters[i]

        filename = "/" + str(counts[letter]).zfill(5) + ".png"

        path = seg_path + letter + "/" + filename
        cv2.imwrite(path, img)
        counts[letter] += 1


            #letter = image[y:y+h, x:x+w]

            #rgb = cv2.cvtColor(letter, cv2.COLOR_GRAY2RGB)
            #pil_img = Image.fromarray(rgb)

            #res = pytesseract.image_to_string(pil_img, config=tess_config)
            #print(res)

            #plt.imshow(rgb)
            #plt.show()
    
    # The second cell is the label matrix
    #labels = output[1]
    # The third cell is the stat matrix
    #stats = output[2]
    # The fourth cell is the centroid matrix
    #centroids = output[3]

    #print(num_labels)
    #print(labels)

   