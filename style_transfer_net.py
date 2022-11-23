import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import style_function as SF

decoder = nn.Sequential(
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

class Style_Transfer_Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Style_Transfer_Net, self).__init__()

        encoder_layers = list(encoder.children())
        self.enc_layer1 = encoder_layers[:3]
        self.enc_layer2 = encoder_layers[3:8]
        self.enc_layer3 = encoder_layers[8:13]
        self.enc_layer4 = encoder_layers[13:]
        self.decoder = decoder

        self.mse_loss = nn.MSELoss()

    def encode(self, input_img):
        feats=[]
        for i in range(4):
            feat = getattr(self, 'enc_layer{:d}'.format(i+1))(input_img)
            feats.append(feat)

        return feats

    
    def forward(self, content_img, style_img, alpha=1.0):
        feats_content = self.encode(content_img)
        feats_style = self.encode(style_img)
        feat_stylized = SF.adaIN(feats_content[-1], feats_style[-1])

        stylized_img = self.decoder(feat_stylized)
        feats_stylized_img = self.encode(stylized_img)

        content_loss = SF.content_loss(feat_stylized, feats_stylized_img[-1])
        style_loss = SF.style_loss(feats_style, feats_stylized_img)

        return content_loss, style_loss
