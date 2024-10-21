
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM, CNNLSTM, CNNTransformerfEncoder
from facemorphic_dataset import FacemorphicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
# import bitsandbytes as bnb
import sys 
import cv2
import torch.nn.functional as F

params = {
    "mode": 'event_rgb',
    "learning_rate": 0.0001,
    "num_epochs": 10000,
    "batch_size_train": 16,
    "batch_size_test": 1,
    "max_seq_len": 50,
    "patch_size": 36,
    'au': 'AU',
    'weight_decay': 0.001
}

dataset_test = FacemorphicDataset('recordings/', split='test', mode=params['mode'], task=params['au'], toy=False, max_seq_len=params['max_seq_len'], use_annot=True, use_cache=True)

test_dataloader = DataLoader(dataset_test, batch_size=params['batch_size_test'], shuffle=True, num_workers=1, drop_last=True)

filt = torch.ones(1, 1, 3, 3, 3)

for batch in test_dataloader:
    #inputs = batch[f'{params["mode"]}_imgs'].cuda().float()/255.0
    inputs_event = batch[f'event_imgs'].cuda().float()/255.0
    inputs_rgb = batch[f'rgb_imgs'].cuda().float()/255.0
    labels = batch['label'].cuda()

    #inp = inputs.squeeze()[None,None,...]
    #out = F.conv3d(inp, filt.cuda(), padding=(1,1,1)).cpu().numpy() > 1


    #print(inputs.shape)
    # for i in range(inputs.shape[1]):
    for i in range(inputs_event.shape[1]):
        #cv2.imshow('img', inputs[0,i,...].cpu().numpy().squeeze())
        #masked = inputs[0,i,...].cpu().numpy().squeeze() * out[0,0,i,...].astype(np.float32).squeeze()
        #cv2.imshow('img', out[0,0,i,...].astype(np.float32).squeeze())

        # copy along the channels
        inputs_event = inputs_event.expand(-1, -1, -1, -1, 3)

        conc = torch.cat((inputs_event, inputs_rgb), 3)
        cv2.imshow('img', conc[0,i,...].cpu().numpy().squeeze())

        #cv2.imshow('img', masked)
        cv2.waitKey(100)

# coords = np.where(inputs.cpu().numpy().squeeze())
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(coords[0], coords[1], coords[2], s=1); plt.show()