import torch
from torch import nn
import torchvision

class EncoderResnet101(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(EncoderResnet101, self).__init__()
        self.enc_image_size = encoded_image_size
     
        resnet_101 = torchvision.models.resnet101(pretrained=True)  

        layers = list(resnet_101.children())[:-1]
        
        self.encoder_net= nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.set_finetune_parameters()

    def forward(self, images):
        out = self.encoder_net(images)  
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)  
        return out

    def set_finetune_parameters(self, fine_tune=True):

        for p in self.encoder_net.parameters():
            p.requires_grad = False
        for c in list(self.encoder_net.children())[4:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
