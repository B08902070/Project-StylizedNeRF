import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import style_function as SF
from VGG import decoder, vgg

class NST_Net(nn.Module):
    def __init__(self, encoder_pretrained_path):
        super(NST_Net, self).__init__()
       
        encoder = vgg.load_state_dict(torch.load(encoder_pretrained_path))
        encoder_layers = list(encoder.children())
        self.enc_layer1 = encoder_layers[:3]
        self.enc_layer2 = encoder_layers[3:8]
        self.enc_layer3 = encoder_layers[8:13]
        self.enc_layer4 = encoder_layers[13:22]

        self.decoder = decoder

        for i in range(1, 5):
            for param in getattr(self, f'enc_layer{i}').parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def load_decoder_state_dict(self, decoder_state_dict):
        self.decoder.load_state_dict(decoder_state_dict)

    def encode(self, input_img):
        feats=[]
        for i in range(4):
            feat = getattr(self, 'enc_layer{:d}'.format(i+1))(input_img)
            feats.append(feat)

        return feats

    
    def forward(self, content_img, style_img, alpha=1.0, return_img_and_feat=False):
        feats_content = self.encode(content_img)
        feats_style = self.encode(style_img)
        feat_stylized = SF.adaIN(feats_content[-1], feats_style[-1])
        feat_stylized = alpha * feat_stylized + (1-alpha) * feats_content

        stylized_img = torch.clamp(self.decoder(feat_stylized), 0, 1)
        feats_stylized_img = self.encode(stylized_img)

        content_loss = SF.content_loss(feat_stylized, feats_stylized_img[-1])
        style_loss = SF.style_loss(feats_style, feats_stylized_img)

        if return_img_and_feat:
            return content_loss, style_loss, style_img, feat_stylized
        else:
            return content_loss, style_loss


