from random import random, choice, randint
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw

class yolo_dataset(Dataset):

    def __init__(self, S, B, C, IMG_SIZE, imgs_list):
        self.imgs_list = imgs_list
        self.S = S
        self.B = B
        self.C = C
        self.img_size = IMG_SIZE
        self.annotations = pd.read_csv('annotations.csv')

        self.transformer = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])

        self.transformer = transforms.Compose([
            transforms.Normalize(mean=(-1., -1., -1.), std=(2., 2., 2.)),
            transforms.ToPILImage()
        ])
    
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        img_pil = Image.open(img_path)
        img_tensor = self.transformer(img_pil)
        annotations = self.annotations[self.annotations['img_path'] == img_path].values

        target_labels = self.preparar_anotacoes(annotations)
        
        return img_tensor, target_labels
    
    def preparar_anotacoes(self, annotations_):
        r"""
        Esta função prepara as anotações no formato target igual ao formato da predição da rede yolo.
        """
        target_labels = torch.zeros((self.S, self.S, 5)) # [S, S, 5] onde 5 -> [p, xc, yc, w, h]

        for img_path, imgw, imgh, xc, yc, w, h in annotations_:
            
            j, i = int(self.S * xc), int(self.S * yc) # índices j, i da célula a qual percence o centro xc, yc, respectivamente
            xc_rel, yc_rel = self.S*xc - j, self.S*yc - i # posição relativa dos centros em comparação à posição x1, y1 da célula
            w_rel, h_rel = w*self.S, h*self.S # tamanhos w e h relativos à célula. 

            if (target_labels[i, j, 0] == 0):
                target_labels[i, j] = torch.tensor([1, xc_rel, yc_rel, w_rel, h_rel])
        
        return target_labels

if __name__ == '__main__':

    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    dataset = yolo_dataset(4, 1, 1, 300, imgs_list)

    img_tensor, bbox_tensor = choice(dataset)
    print (f'img_tensor.shape: {img_tensor.shape}, bbox_tensor.shape: {bbox_tensor.shape}')