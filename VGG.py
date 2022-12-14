import numpy as np
import torch.nn as nn

vgg = nn.Sequential(
                    # layer 1
                    nn.Conv2d(3, 3, kernel_size=1, stride = 1),
                    nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                        
                    # layer 2
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                        
                    # layer 3
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                        
                    # layer 4
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),


                    # the rest are not used, but for fitting pretrained data
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                    nn.ReLU(inplace=True),
                   )

decoder = nn.Sequential(
                        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect'),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode =  'reflect')
                       )