import torch
from torch import nn


S = 4 # número de grids
C = 1 # número de classes
B = 1 # número de bbox predito por célula
IMG_SIZE = 300 # imagem para redimensionar e passar pela rede neural