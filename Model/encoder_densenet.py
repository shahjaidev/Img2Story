import torch
from torch import nn
import torchvision
import json 
import numpy as np
import bcolz 
import pickle


class EncoderDensenet(nn.Module):

    def __init__(self, encoded_image_size=8):
        super(EncoderDensenet, self).__init__()
        self.enc_image_size = encoded_image_size
        

        densenet= torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        layers = list(densenet.children())[:-1]
        self.encoder_net = nn.Sequential(*layers,nn.Conv2d(1024, 2048, 1)  ) 
        #self.conv1 = nn.Conv2d(1280, 2048, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.set_finetune_parameters(True)

    def forward(self, images):
        out = self.encoder_net(images)  
        #out = self.conv1(out)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # 2048 filters 
        out = out.permute(0, 2, 3, 1)  
        return out

    def set_finetune_parameters(self, fine_tune=True):

        for p in self.encoder_net.parameters():
            p.requires_grad = False
        
        children= list(self.encoder_net.children())
        for c in children[0][10:]:
            for p in c.parameters():
                p.requires_grad = True

        for p in children[1].parameters():
            p.requires_grad = True
