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

    
class ModelEncoderDecoder(nn.Module):
    def __init__(self, num_classes=24, feat_dim=64, hidden_dim=256, mlp_dim=128, image_size=720, patch_size=72, seq_len=75):
        super(ModelEncoderDecoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        self.seq_len = seq_len
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.encoder = ViTNoCLS(image_size=self.image_size,
                           patch_size=self.patch_size,
                           dim=self.feat_dim,
                           depth=1,
                           heads=4,
                           mlp_dim=self.mlp_dim,
                           channels=1,
                           dropout=0.2,
                           emb_dropout=0.2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.feat_dim, nhead=4, dim_feedforward=self.mlp_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.cls_token_decoder = nn.Parameter(torch.randn(1, 1, self.feat_dim))
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.feat_dim))
        self.fc1 = nn.Linear(self.feat_dim*self.seq_len, self.num_classes)        
        #self.fc1 = nn.Linear(self.feat_dim, self.num_classes)        
    
    def forward(self, x):

        x = F.interpolate(x, size=(x.shape[2], 224, 224))

        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]

        x = self.encoder(x.view(-1, x.shape[2], x.shape[3], x.shape[4])) # encoded inputs
        x = x.view(b, seq_length, x.shape[1], x.shape[2])
        
        cls_tokens = repeat(self.cls_token_decoder, '1 1 d -> b 1 d', b = b)
        M = torch.cat((cls_tokens, x[:, 0, ...]), dim=1)

        x += self.pos_embedding[:, :(self.num_patches)].expand(x.shape)

        tokens = []
        tokens.append(torch.clone(M[:,0,...]))

        for i in range(1, seq_length):
            M += self.pos_embedding[:, :(self.num_patches + 1)]
            M = self.decoder(M, x[:,i,...])
            tokens.append(torch.clone(M[:,0,...]))
        #out = self.fc1(M[:,0,...])
        out = self.fc1(torch.cat(tokens, dim=1))
        # out = self.fc1(tokens[-1])
        return out
    



class ModelEncoder(nn.Module):
    def __init__(self, num_classes=24, feat_dim=64, hidden_dim=64, image_size=720, patch_size=72):
        super(ModelEncoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.encoder = ViT(image_size=self.image_size,
                           patch_size=self.patch_size,
                           dim=self.feat_dim,
                           depth=1,
                           heads=1,
                           mlp_dim=32,
                           channels=1,
                           dropout=0.2,
                           emb_dropout=0.2)
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.fc1 = nn.Linear(self.feat_dim*75, self.num_classes)
        
    
    def forward(self, x):
        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        #seq_length = x.shape[1]

        x = self.encoder(x.view(-1, x.shape[2], x.shape[3], x.shape[4])) # encoded inputs
        x = rearrange(x, '(b t) d -> b (t d)', t=75)
        
        out = self.fc1(x)
        return out
    

class ResnetLSTM(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256, image_size=720, pretrain=False):

        super(ResnetLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.pretrain = pretrain
        self.resnet = torchvision.models.resnet18(pretrained=self.pretrain)
        self.resnet.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        
    def forward(self, x):
        # extend input to have 3 channels without allocating memory
        x = x.expand(-1, -1, 3, -1, -1)        

        # resize images to (180, 180)
        x = F.interpolate(x, size=(3, 224, 224))

        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]
        x = self.resnet(x.view(-1, x.shape[2], x.shape[3], x.shape[4])) # encoded inputs
        x = x.view(b, seq_length, x.shape[1])
        x, _ = self.lstm(x)
        x = x[:,-1,:] # get last time step
        out = self.fc(x)
        return out


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256, image_size=720):
        super(CNNLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(6, 9, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(9, 12, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(108, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        
    def forward(self, x):
        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 108)
        x = F.relu(self.fc1(x))
        x = x.view(b, seq_length, x.shape[1])
        x, _ = self.lstm(x)
        x = x[:,-1,:] # get last time step
        out = self.fc2(x)
        return out
    

class CNNTransformerfEncoder(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256, in_channels=1):
        super(CNNTransformerfEncoder, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=5, stride=1)
        # batch norm
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 9, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(9)
        self.conv4 = nn.Conv2d(9, 12, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1452, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.transformer_encoder = model_layers.TransformerEncoderWrapper(self.hidden_dim, 4, 64, 1, 0.2, num_cls_tokens=1)
        
    def forward(self, x):
        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = rearrange(x, '(b s) c h w -> (b s) (c h w)', b=b, s=seq_length)
        x = F.relu(self.fc1(x))
        x = rearrange(x, '(b s) f -> s b f', b=b, s=seq_length) # DO NOT USE VIEW!
        _, x = self.transformer_encoder(x)
        x = x.view(b, self.hidden_dim)
        out = self.fc2(x)
        return out
    
class CNNTransformerAlpha(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256, in_channels=1):
        super(CNNTransformerAlpha, self).__init__()
        self.cnn_transformer = CNNTransformerfEncoder(num_classes=num_classes, hidden_dim=hidden_dim, in_channels=in_channels)
        self.fc_alpha = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        
        x = self.cnn_transformer.pool(F.relu(self.cnn_transformer.bn1(self.cnn_transformer.conv1(x))))
        x = self.cnn_transformer.pool(F.relu(self.cnn_transformer.bn2(self.cnn_transformer.conv2(x))))
        x = self.cnn_transformer.pool(F.relu(self.cnn_transformer.bn3(self.cnn_transformer.conv3(x))))
        x = self.cnn_transformer.pool(F.relu(self.cnn_transformer.bn4(self.cnn_transformer.conv4(x))))

        x = rearrange(x, '(b s) c h w -> (b s) (c h w)', b=b, s=seq_length)
        x = F.relu(self.cnn_transformer.fc1(x))
        x = rearrange(x, '(b s) f -> s b f', b=b, s=seq_length) # DO NOT USE VIEW!
        x_seq, x = self.cnn_transformer.transformer_encoder(x)
        x = x.view(b, self.cnn_transformer.hidden_dim)
        out = self.cnn_transformer.fc2(x)
        alpha_out = self.fc_alpha(F.relu(x_seq))
        alpha_out = alpha_out.permute(1, 0, 2)
        return out, alpha_out


class TwoStreamCNNTransformerEncoder(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256):
        super(TwoStreamCNNTransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rgb_stream = CNNTransformerfEncoder(num_classes=num_classes, hidden_dim=hidden_dim, in_channels=3)
        self.event_stream = CNNTransformerfEncoder(num_classes=num_classes, hidden_dim=hidden_dim, in_channels=1)
        self.rgb_stream.fc2 = nn.Identity()
        self.event_stream.fc2 = nn.Identity()
        self.fc = nn.Linear(2*self.hidden_dim, self.num_classes)

    def forward(self, x_event, x_rgb):
        x_event = self.event_stream(x_event)
        x_rgb = self.rgb_stream(x_rgb)
        x = torch.cat((x_event, x_rgb), dim=1)
        out = self.fc(x)
        return out

class TransformerAlphaClassifier(nn.Module):
    def __init__(self, num_classes=24):
        super(TransformerAlphaClassifier, self).__init__()
        self.num_classes = num_classes
        self.transformer_encoder = model_layers.TransformerEncoderWrapper(32, 4, 64, 1, 0.2, num_cls_tokens=1)
        self.fc = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        _, x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
    

class LSTMAlphaClassifier(nn.Module):
    def __init__(self, num_classes=24):
        super(LSTMAlphaClassifier, self).__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:] # get last time step
        x = self.fc(x)
        return x
    

class ResnetEncoderDecoder(nn.Module):
    def __init__(self, num_classes=24, feat_dim=64, hidden_dim=256, mlp_dim=128, image_size=720, seq_len=75):
        super(ResnetEncoderDecoder, self).__init__()

        self.image_size = image_size
        self.patch_size = 1
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        self.seq_len = seq_len

        self.resnet = model_layers.ResnetOnlyConv()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.encoder = ViTNoCLS(image_size=self.image_size,
                           patch_size=self.patch_size,
                           dim=self.feat_dim,
                           depth=1,
                           heads=4,
                           mlp_dim=self.mlp_dim,
                           channels=512,
                           dropout=0.2,
                           emb_dropout=0.2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.feat_dim, nhead=4, dim_feedforward=self.mlp_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.cls_token_decoder = nn.Parameter(torch.randn(1, 1, self.feat_dim))
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.feat_dim))
        self.fc1 = nn.Linear(self.feat_dim*self.seq_len, self.num_classes)        
        #self.fc1 = nn.Linear(self.feat_dim, self.num_classes)        
    
    def forward(self, x):
        # x.shape - (batch, time, C, H, W)
        b = x.shape[0]
        seq_length = x.shape[1]

        # extend input to have 3 channels without allocating memory
        x = x.expand(-1, -1, 3, -1, -1)        

        # resize images to (180, 180)
        x = F.interpolate(x, size=(3, 224, 224))

        x = x.view(b*seq_length, x.shape[2], x.shape[3], x.shape[4])
        x = self.resnet(x)

        x = self.encoder(x) # encoded inputs
        x = x.view(b, seq_length, x.shape[1], x.shape[2])
        
        cls_tokens = repeat(self.cls_token_decoder, '1 1 d -> b 1 d', b = b)
        M = torch.cat((cls_tokens, x[:, 0, ...]), dim=1)

        x += self.pos_embedding[:, :(self.num_patches)].expand(x.shape)

        tokens = []
        tokens.append(torch.clone(M[:,0,...]))

        for i in range(1, seq_length):
            M += self.pos_embedding[:, :(self.num_patches + 1)]
            M = self.decoder(M, x[:,i,...])
            tokens.append(torch.clone(M[:,0,...]))
        #out = self.fc1(M[:,0,...])
        out = self.fc1(torch.cat(tokens, dim=1))
        # out = self.fc1(tokens[-1])
        return out



class ResnetTwoStream(nn.Module):
    def __init__(self, num_classes=24, feat_dim=64, feat_dim2= 32, hidden_dim=256, mlp_dim=128, image_size=720, seq_len=75):
        super(ResnetTwoStream, self).__init__()

        self.image_size = image_size
        self.patch_size = 1
        self.seq_len = seq_len

        self.resnet_event = model_layers.ResnetOnlyConv(cutoff_layer=3, pretrained=True)
        self.resnet_rgb = model_layers.ResnetOnlyConv(cutoff_layer=3, pretrained=True)
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.feat_dim2 = feat_dim2
        self.mlp_dim = mlp_dim

        self.encoder_rgb = Transformer(dim=512, depth=1, heads=4, dim_head=64, mlp_dim=self.mlp_dim, dropout=0.2)
        self.encoder_event = Transformer(dim=512, depth=1, heads=4, dim_head=64, mlp_dim=self.mlp_dim, dropout=0.2)
    
        self.cls_token_event = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token_rgb = nn.Parameter(torch.randn(1, 1, 512))

        self.num_patches = 7*7
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 2, 512))
        self.fc1 = nn.Linear(1024, self.feat_dim2)
        self.fc2 = nn.Linear(self.feat_dim2, self.num_classes)

    
    def forward(self, x_event, x_rgb):
        # x.shape - (batch, time, C, H, W)
        b = x_event.shape[0]
        seq_length = x_event.shape[1]

        # extend input to have 3 channels without allocating memory
        x_event = x_event.expand(-1, -1, 3, -1, -1)        

        # resize images to (180, 180)
        x_event = F.interpolate(x_event, size=(3, 224, 224))
        x_rgb = F.interpolate(x_rgb, size=(3, 224, 224))

        # Two stream Resnet backbone - EVENT
        x_event = x_event.view(b*seq_length, x_event.shape[2], x_event.shape[3], x_event.shape[4])
        x_event = self.resnet_event(x_event)
        x_event = x_event.view(b, seq_length, x_event.shape[1], x_event.shape[2]*x_event.shape[2])
        # Two stream Resnet backbone - RGB
        x_rgb = x_rgb.view(b*seq_length, x_rgb.shape[2], x_rgb.shape[3], x_rgb.shape[4])
        x_rgb = self.resnet_event(x_rgb)
        x_rgb = x_rgb.view(b, seq_length, x_rgb.shape[1], x_rgb.shape[2]*x_rgb.shape[2])

        # Adapt cls tokens to batch size
        cls_tokens_event = self.cls_token_event.expand(b, 1, self.cls_token_event.shape[-1])
        cls_tokens_rgb = self.cls_token_rgb.expand(b, 1, self.cls_token_rgb.shape[-1])

        for i in range(seq_length):

            x_event_i = x_event[:, i, ...]
            x_rgb_i = x_rgb[:, i, ...]

            # concatenate cls tokens
            x_event_i = torch.cat((x_event_i.permute(0,2,1), cls_tokens_rgb, cls_tokens_event), dim=1)
            x_rgb_i = torch.cat((x_rgb_i.permute(0,2,1), cls_tokens_event, cls_tokens_rgb), dim=1)

            # add positional encoding
            x_event_i += self.pos_embedding
            x_rgb_i += self.pos_embedding
            
            # Recurrent encoder
            x_event_i = self.encoder_event(x_event_i) # encoded inputs
            x_rgb_i = self.encoder_rgb(x_rgb_i)

            # separate outputs for cls tokens and src tokens
            cls_tokens_event = x_event_i[:, -1:, ...] # -1: -> index last and keep dim
            cls_tokens_rgb = x_rgb_i[:, -1:, ...]


        # concatenate cls tokens for event and rgb
        x = torch.cat((cls_tokens_event, cls_tokens_rgb), dim=2).squeeze()

        # classification mlp
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ViVit_Model(nn.Module):
    def __init__(self,image_size=224, frames=50, image_patch_size=16, frame_patch_size=4, num_classes=24, dim=512, spatial_depth=2, temporal_depth=2, heads=4, mlp_dim=128, channels=3):
        super(ViVit_Model, self).__init__()
        self.vivit = ViViT(image_size=image_size,
                           frames=frames,
                           image_patch_size=image_patch_size,
                           frame_patch_size=frame_patch_size,
                           num_classes=num_classes,
                           dim=dim,
                           spatial_depth=spatial_depth,
                           temporal_depth=temporal_depth,
                           heads=heads,
                           mlp_dim=mlp_dim,
                           channels=channels)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.vivit(x)
        return out
