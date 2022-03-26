import torch
from torch import nn
from utils import calculate_ious
import pandas as pd

class YOLO_LOSS(nn.Module):

    def __init__(self, S, B, C, IMG_SIZE):
        super(YOLO_LOSS, self).__init__()
        r"""
        
        """
        self.S = S
        self.B = B
        self.C = C
        self.img_size = IMG_SIZE
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        r"""
        Args: 
            predictions: vem no formato flatten. [N, S*S*5]
            targets: vem no formato [S, S, 5]
        
        Returns: 
            loss: o somatório de todas as perdas, ponderadas.
        """
        # aqui faz o reshape do predictions igual ao shape do target.
        predictions = predictions.reshape(targets.shape)

        # A variável exists_box é a que possui em qual célula está pre
        # sente o objeto.
        exists_box = targets[:, :, :, 0].unsqueeze(3)
        no_exists_box = 1 - exists_box

        # ======================= CALCULO PERDA PARA COORDENADAS =============================
        # Aqui fazemos a multiplicação e pegamos as predições
        # somente das células que de fato são responsáveis pelo objeto
        box_predictions = exists_box * predictions[:, :, :, 1:]
        box_targets = exists_box * targets[:, :, :, 1:]

        # De acordo com a função perda do paper, o somatório de w e h
        # é feito pelas suas raizes quadradas. Assim vamos alterar somente
        # esses itens.

        # aqui fazemos:                torch.sign para respeitar o sinal    # raiz quadrada do absoluto + 1e-6 para não dar erro
        box_predictions[:,:,:,2:] = torch.sign(box_predictions[:,:,:,2:]) * torch.sqrt(torch.abs(box_predictions[:,:,:,2:] + 1e-6))
        box_targets[:,:,:,2:] = torch.sign(box_targets[:,:,:,2:]) * torch.sqrt(torch.abs(box_targets[:,:,:,2:] + 1e-6))
        # para esse segundo não é necessário, mas fez apenas para manter o padrão

        box_loss = self.mse(box_predictions.reshape(-1, 4), box_targets.reshape(-1, 4))
        # ======================= CALCULO PERDA PARA COORDENADAS =============================

        # Como este problema só tem uma classe, então não temos cálculo de perdas
        # para classes. Assim, vamos direto para última linha da função perda

        # ====================== CALCULO PARA PROBABILIDADE DE DETECAÇÃO DE OBJETO ===========
        prob_exists_prediction = exists_box * predictions[:,:,:,0].unsqueeze(3)
        prob_exists_target = exists_box * targets[:,:,:,0].unsqueeze(3)
        prob_exists_loss = self.mse(
            prob_exists_prediction.reshape(-1, 1), 
            prob_exists_target.reshape(-1, 1)
        )
        # ====================== CALCULO PARA PROBABILIDADE DE DETECAÇÃO DE OBJETO ===========

        # ================= CALCULO PARA PROBABILIDADE DE NÃO DETECAÇÃO DE OBJETO ============
        prob_no_exists_prediction = no_exists_box * predictions[:,:,:,0].unsqueeze(3)
        prob_no_exists_target = no_exists_box * targets[:,:,:,0].unsqueeze(3)
        prob_noobj_loss = self.mse(
            prob_no_exists_prediction.reshape(-1, 1),
            prob_no_exists_target.reshape(-1, 1)
        )
        # ================= CALCULO PARA PROBABILIDADE DE NÃO DETECAÇÃO DE OBJETO ============
        
        loss = (
            box_loss * self.lambda_coord
            + prob_exists_loss
            + prob_noobj_loss * self.lambda_noobj
        )

        return loss


if __name__ == '__main__':

    from model import YOLO, S, C, B, IMG_SIZE
    from dataset import yolo_dataset
    from torch.utils.data import DataLoader

    model = YOLO(S, C, B, IMG_SIZE)
    yolo_loss = YOLO_LOSS(S, B, C, IMG_SIZE)
    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    dataset = yolo_dataset(S, B, C, IMG_SIZE, imgs_list)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    imgs_tensor, target_tensor = next(iter(dataloader))
    predictions = model(imgs_tensor)

    loss = yolo_loss(predictions, target_tensor)
    print (f'loss: {loss}')