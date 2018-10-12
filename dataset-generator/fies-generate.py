import numpy as np
import random
import string
import sys

from PIL import Image, ImageDraw, ImageFont

# fonte usada (bistream vera)
font = ImageFont.truetype('Vera.ttf', size=50)

# caminho
path = "../dataset/raw/"

def rndPointDisposition(dx, dy):
    """Return random disposition point."""
    x = int(random.uniform(-dx, dx))
    y = int(random.uniform(-dy, dy))
    return (x, y)

def quadPoints(size, disp1, disp2):
    """Return points for QUAD transformation."""
    w, h = size
    x1, y1 = disp1
    x2, y2 = disp2

    return (
        x1,    -y1,
        -x1,    h + y2,
        w + x2, h - y2,
        w - x2, y1)
    
def rndLineTransform(image):
    """Randomly morph Image object with drawn line."""
    w, h = image.size

    # default: 0.3 0.5
    dx = w * random.uniform(0.2, 0.4)
    dy = h * random.uniform(0.2, 0.4)

    x1, y1 = [abs(z) for z in rndPointDisposition(dx, dy)]
    x2, y2 = [abs(z) for z in rndPointDisposition(dx, dy)]

    quad = quadPoints((w, h), (x1, y1), (x2, y2))

    return image.transform(image.size, Image.QUAD,
                            data=quad, resample=Image.BICUBIC, fill=1)

def deform_image(image):
    transformed_image = rndLineTransform(image)

    new_image = Image.new('RGBA', (190, 80), color=(255, 255, 255))
    new_image.paste(transformed_image, transformed_image)

    return new_image

def draw_cross(ctx, x, y):
    ctx.point((x, y), 'black')
    ctx.point((x+1, y), 'black')
    ctx.point((x-1, y), 'black')
    ctx.point((x, y+1), 'black')
    ctx.point((x, y-1), 'black')

def draw_random_cross(ctx):
    x1 = random.randint(1, 189)
    y1 = random.randint(1, 79)

    draw_cross(ctx, x1, y1)

def draw_random_line(ctx):
    x1 = random.randint(0, 190)
    y1 = random.randint(0, 80)

    x2 = random.randint(0, 190)
    y2 = random.randint(0, 80)
    ctx.line((x1, y1, x2, y2), 'black')

def draw_random_stuff(ctx):
    num_crosses = random.randint(80, 90)

    for i in range(num_crosses):
        draw_random_cross(ctx)   
    
    num_lines = random.randint(5, 7)

    for i in range(num_lines):
        draw_random_line(ctx)

def gen_captcha(text):
    # cria uma imagem branca de 190x80
    image = Image.new('RGBA', (190, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # passo 1: desenha pontos e linhas aleatorias sem deformação e o texto
    draw_random_stuff(draw)
    draw.text((40, 20), text, fill='black', font=font)
    del draw

    # passo 2: transforma a imagem
    image = deform_image(image)

    # passo 3: repetir passo 1
    draw = ImageDraw.Draw(image)
    draw_random_stuff(draw)
    del draw

    return image

def gen_string(size=4, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if len(sys.argv) == 1:
    print("ha")
else:
    num = int(sys.argv[1])
    print(f"Gerando {num} captchas")

    for i in range(num):
        if i % 10 == 0:
            print(f"{str(i)} CAPTCHAS gerados")
        text = gen_string()
        image = gen_captcha(text)
        filename = text + ".png"
        image.save(path + filename)