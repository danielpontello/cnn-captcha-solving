import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import string

from PIL import Image

kernel = (3, 3)
level = 2
tess_config = "-oem 0 -c tessedit_char_whitelist=" + string.ascii_lowercase + string.digits

print(tess_config)

for i in range(1, 185):
    num = str(i).zfill(5)
    image = cv2.imread("raw-images/" + num + ".png", 0)

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

    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]

        if a > 50:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            letter = image[y:y+h, x:x+w]

            rgb = cv2.cvtColor(letter, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb)

            res = pytesseract.image_to_string(pil_img, config=tess_config)
            print(res)

            plt.imshow(rgb)
            plt.show()
    
    # The second cell is the label matrix
    #labels = output[1]
    # The third cell is the stat matrix
    #stats = output[2]
    # The fourth cell is the centroid matrix
    #centroids = output[3]

    #print(num_labels)
    #print(labels)

   