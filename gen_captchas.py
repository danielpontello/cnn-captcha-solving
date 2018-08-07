from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import string
import random
import numpy as np

def generate(width,height,num_char):
    characters = string.digits
    rand_str = ''.join(random.sample(characters,num_char))
    generator = ImageCaptcha(width=width, height=height)
    image = generator.generate_image(rand_str)
    return (image,rand_str)

save_dir = 'captchas/'
labels = []

for i in range(5000):
    if i % 100 == 0:
        print(i, "de 5000 captchas gerados")
    
    img, char = generate(140,80,4)
    plt.imsave(save_dir+str(i) + ".png",np.array(img))
    labels.append(char)
with open(save_dir + 'labels.txt','w') as f:
    for lb in labels:
        f.write('%s\n' % lb)