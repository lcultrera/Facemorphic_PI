import os
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from tqdm import tqdm
from vit import ViT, ViTNoCLS, pair, Transformer
from einops import rearrange, repeat
import model_layers
from vit_pytorch.vivit import ViT as ViViT
import re
from model_layers import ResnetOnlyConv


''' CODES
convolutional layer: 'conv_IN_OUT_KSz_KSt_PSz_PSt' (int, int, int, int, int, int)
linear layer: 'fc_IN_OUT_DROPOUT' (int, int, float)
relu: 'relu' (None)
flatten: 'flatten' (None)
resnet: 'resnet_RESTYPE_CUTOFF_PRETRAINED' (int, int, bool)
'''

class LayerFactory:

    def get_layers(self, layer_codes):
        modules = []
        for layer_code in layer_codes:
            modules.append(self.__get_layer(layer_code))
        return nn.Sequential(*modules)


    def __get_layer(self, layer_code):
        if layer_code.startswith('conv_'):
            # 3D Conv layer
            conv_params = re.match(r'conv_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', layer_code).groups()
            return self.__get_conv_block(*[int(x) for x in conv_params])
        elif layer_code.startswith('flatten'):
            # Flatten layer
            return self.__get_flatten()
        elif layer_code.startswith('relu'):
            # ReLU layer
            return self.__get_relu()
        elif layer_code.startswith('fc_'):
            # Linear layer
            fc_params = re.match(r'fc_(\d+)_(\d+)_(\d+)', layer_code).groups()
            return self.__get_linear_block(int(fc_params[0]), int(fc_params[1]), float(fc_params[2]))
        elif layer_code.startswith('resnet'):
            # Resnet backbone
            resnet_params = re.match(r'resnet_(\d+)_(\d+)_(\d)', layer_code).groups()
            return self.__get_resnet(int(resnet_params[0]), int(resnet_params[1]), bool(resnet_params[2]))

    def __get_conv_block(self, ch_in, ch_out, kernel_size, stride,
                         pool_size, pool_stride):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, pool_stride)
        )
    
    def __get_flatten(self):
        return nn.Flatten()

    def __get_relu(self):
        return nn.ReLU()

    def __get_linear_block(self, feat_in, feat_out, dropout):
        return nn.Sequential(
            nn.Linear(feat_in, feat_out),
            nn.Dropout(dropout)
        )
    
    def __get_resnet(self, resnet_type, cutoff, pretrained):
        assert resnet_type in [18, 34, 50, 101, 152]
        if resnet_type != 18:
            raise NotImplementedError("Only ResNet-18 is supported for now")
        return ResnetOnlyConv(cutoff_layer=cutoff, pretrained=pretrained, resnet_type=resnet_type)
    

factory = LayerFactory()
m = factory.get_layers(['resnet_18_4_1', 'conv_3_64_3_1_2_2', 'conv_64_128_5_1_2_1', 'flatten', 'fc_128_256_0.1', 'relu', 'fc_256_10_0'])
print(m)