import torch
from torch import nn
import torchvision
from torchvision.models.squeezenet import squeezenet1_0
import json 
import numpy as np
import bcolz 
import pickle


class EncoderSqueezenet(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(EncoderSqueezenet, self).__init__()
        self.enc_image_size = encoded_image_size
        
        squeezenet = squeezenet1_0(pretrained=True)
        layers = list(squeezenet.children())[:-1]
        self.encoder_net = nn.Sequential(*layers,nn.Conv2d(512, 2048, 1) ) #nn.BatchNorm2d(num_features=2048,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True )

        #self.conv1 = nn.Conv2d(1280, 2048, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        out = self.encoder_net(images)  
        #out = self.conv1(out)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # 2048 filters 
        out = out.permute(0, 2, 3, 1)  
        return out

    def fine_tune(self, fine_tune=True):

        for p in self.encoder_net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        children= list(self.encoder_net.children())
        for c in children[0]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        for p in children[1].parameters():
            p.requires_grad = fine_tune
            