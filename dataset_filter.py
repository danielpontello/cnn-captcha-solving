import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pytesseract
import random
import string

from PIL import Image

kernel = (3, 3)
level = 2
tess_config = "-oem 0 -c tessedit_char_whitelist=" + string.ascii_lowercase + string.digits

print(tess_config)
images = os.listdir("captcha-dataset")

image = cv2.imread("captcha-dataset/" + random.choice(images), 0)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.subplot(5, 1, 1)
plt.imshow(image_rgb)

# blur
k = np.ones((7, 7),np.float32)/25
blurred = cv2.GaussianBlur(image, (5,5), 0)
#blurred = cv2.filter2D(image,-1,k)
#blurred = image
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
plt.subplot(5, 1, 2)
plt.imshow(blurred_rgb)

# threshold
ret, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)
thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.subplot(5, 1, 3)
plt.imshow(thresh_rgb)

# erode
erode = cv2.erode(thresh, kernel, iterations = 4)
erode_rgb = cv2.cvtColor(erode, cv2.COLOR_GRAY2RGB)
plt.subplot(5, 1, 4)
plt.imshow(erode_rgb)

# dilate
dilate = cv2.dilate(erode, kernel, iterations = 0)
dilate_rgb = cv2.cvtColor(dilate, cv2.COLOR_GRAY2RGB)
plt.subplot(5, 1, 5)
plt.imshow(dilate_rgb)
plt.show()


   