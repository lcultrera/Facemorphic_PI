import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float=0.5, num_cls_tokens=0):
        """
        Arguments:
            d_model: int, the number of expected features in the input
            nhead: int, the number of heads in the multiheadattention models
            d_hid: int, the dimension of the feedforward network model
            nlayers: int, the number of sub-encoder-layers in the encoder
            dropout: float, the dropout value
            num_cls_tokens: int, the number of cls tokens to prepend to the input
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = None
        if num_cls_tokens:
            self.cls_tokens = torch.nn.Parameter(torch.randn(num_cls_tokens, 1, d_model))
    
    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, feat_dim]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, feat_dim]``
            cls Tensor of shape ``[num_cls_tokens, batch_size, feat_dim]``
        """

        seq_len, batch_size, feat_dim = src.shape

        # concatenate cls tokens to src
        if self.num_cls_tokens:
            src = torch.cat([self.cls_tokens.expand(self.num_cls_tokens, batch_size, feat_dim), src], dim=0)

        #src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        src = self.transformer_encoder(src)

        # separate outputs for cls tokens and src tokens
        if self.num_cls_tokens:
            cls_output = src[:self.num_cls_tokens, ...].squeeze()
            src_output = src[self.num_cls_tokens:, ...].squeeze()
            return src_output, cls_output
        else:
            return src, None


class ResnetOnlyConv(nn.Module):
    def __init__(self, cutoff_layer=4, pretrained=True, resnet_type=18):
        super(ResnetOnlyConv, self).__init__()
        #self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = torchvision.models.__dict__[f'resnet{resnet_type}'](pretrained=pretrained)
        self.resnet.fc = nn.Identity()
        self.cutoff_layer = cutoff_layer
        if self.cutoff_layer < 3:
            self.resnet.layer4 = nn.Identity()
        if self.cutoff_layer < 2:
            self.resnet.layer3 = nn.Identity()
        if self.cutoff_layer < 1:
            self.resnet.layer2 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #x = self.resnet.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        return x