from operator import index
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from random import random, choice, randint
from argparse import ArgumentParser
import os
from tqdm import tqdm

def random_color(clara=True):
    (valor_min, valor_max) = (150,255) if clara else (0,150)
    return (randint(valor_min, valor_max), randint(valor_min, valor_max), randint(valor_min, valor_max))

def create_image(n_objects = 1):
    imgw, imgh = randint(200, 300), randint(200, 300)
    img_pil = Image.new('RGB', size=(imgw, imgh), color='white')
    draw = ImageDraw.Draw(img_pil)

    for _ in range(40):
        x1, y1, x2, y2 = randint(0, imgw), randint(0, imgh), randint(0, imgw), randint(0, imgh)
        draw.line([x1, y1, x2, y2], fill=random_color(), width=1)
    
    bboxes = []
    for n in range(n_objects):
        w, h = randint(30, 60), randint(30, 60)
        x1, y1 = randint(0, imgw-w), randint(0, imgh-h)
        x2, y2 = x1+w, y1+h
        xc, yc = x1+w/2, y1+h/2

        draw.ellipse([x1, y1, x2, y2], fill=random_color(False))
        bboxes.append([imgw, imgh, xc/imgw, yc/imgh, w/imgw, h/imgh])

    return img_pil, bboxes

def create_dataset(n_images):

    if (not(os.path.exists('./imgs'))):
        os.mkdir('./imgs')
    
    annotations = []
    for k in tqdm(range(n_images), ncols=50):
        img_path = f'./imgs/img_{k:04}.jpg'

        img_pil, bboxes = create_image(choice([1, 2]))
        img_pil.save(img_path)

        for imgw, imgh, xc, yc, w, h in bboxes:
            annotations.append([img_path, imgw, imgh, xc, yc, w, h])
    
    df = pd.DataFrame(annotations, columns=['img_path', 'imgw', 'imgh', 'xc', 'yc', 'w', 'h'])
    df.to_csv('annotations.csv', index=False)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n', help='Numero de imagens')

    args = parser.parse_args()

    create_dataset(int(args.n))