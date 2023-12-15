import numpy
import torch
from torch import nn


class DAE(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.encoder_layer1 = nn.Linear(in_dim, 32)
        self.encoder_layer2 = nn.Linear(32, 16)
        self.encoder_layer3 = nn.Linear(16, 16)
        self.encoder_layer4 = nn.Linear(16, 8)

        self.decoder_layer1 = nn.Linear(8, 16)
        self.decoder_layer2 = nn.Linear(16, 16)
        self.decoder_layer3 = nn.Linear(16, 32)
        self.decoder_layer4 = nn.Linear(32, in_dim)
        
        self.encoder_layer5 = nn.Linear(in_dim, 32)
        self.encoder_layer6 = nn.Linear(32, 16)
        self.encoder_layer7 = nn.Linear(16, 16)
        self.encoder_layer8 = nn.Linear(16, 8)


    def forward(self,x):
        x_2 = x.clone()
        x_2_0 = x_2.clone()
        x_2 = self.encoder_layer1(x_2)
        x_2 = self.encoder_layer2(x_2)
        x_2 = self.encoder_layer3(x_2)
        x_2 = self.encoder_layer4(x_2)
        z_1 = x_2.clone()
        z_out = z_1.clone()
        x_2 = self.decoder_layer1(x_2)
        x_2 = self.decoder_layer2(x_2)
        x_2 = self.decoder_layer3(x_2)
        x_2 = self.decoder_layer4(x_2)
        #x_2 = torch.sigmoid(x_2)
        x_2_1 = x_2.clone()
        x_2 = self.encoder_layer5(x_2)
        x_2 = self.encoder_layer6(x_2)
        x_2 = self.encoder_layer7(x_2)
        x_2 = self.encoder_layer8(x_2)
        #x_2 = torch.sigmoid(x_2)
        z_2 = x_2.clone()
        x_dist = torch.pow((x_2_1 - x_2_0) * (x_2_1 - x_2_0), 0.5)
        z_dist = torch.pow((z_1 - z_2)*(z_1 - z_2),0.5)
        return x_2_1,z_dist,x_dist,z_out,z_1,z_2