import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')

S = 4 # número de grids
C = 1 # número de classes
B = 1 # número de bbox predito por célula
IMG_SIZE = 300 # imagem para redimensionar e passar pela rede neural

class YOLO(nn.Module):

    def __init__(self, S, C, B, IMG_SIZE):

        super(YOLO, self).__init__()
        self.S = S
        self.C = C
        self.B = B
        self.img_size = IMG_SIZE

        backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )

        input_test = torch.rand((1, 3, self.img_size, self.img_size))
        output = backbone(input_test)
        in_features_linear = output.shape[-1]

        self.net = nn.Sequential(
            backbone, 
            nn.Linear(in_features=in_features_linear, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=(self.S * self.S * (1 + 4)))
        )
    
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':

    model = YOLO(S, C, B, IMG_SIZE)
    input = torch.rand((2, 3, IMG_SIZE, IMG_SIZE))
    output = model(input)
    print (f'output.shape: {output.shape}')