import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import style_function as SF

class Style_Transfer_Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Style_Transfer_Net, self).__init__()

        encoder_layers = list(encoder.children())
        self.enc_layer1 = encoder_layers[:3]
        self.enc_layer2 = encoder_layers[3:8]
        self.enc_layer3 = encoder_layers[8:13]
        self.enc_layer2 = encoder_layers[13:]
        self.decoder = nn.Sequential(
                        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(factor=2, mode='nearest'),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(factor=2, mode='nearest'),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(factor=2, mode='nearest'),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect')
                       )

        self.mse_loss = nn.MSELoss()

    def forward(self, x, alpha=1.0):

