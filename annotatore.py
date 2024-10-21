from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM
from facemorphic_dataset import FacemorphicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import sys 
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F


params = {
    "mode": 'event',
    "learning_rate": 0.0001,
    "num_epochs": 1000,
    "batch_size_train": 8,
    "batch_size_test": 2,
    "max_seq_len": 999999,
    "patch_size": 36,
    'au': 'AU_NOHEAD'
}
dataset_train = FacemorphicDataset('recordings/', split='train', mode=params['mode'], au_labels=['AU_7'], task=params['au'], toy=False, max_seq_len=params['max_seq_len'], clean_cache=False, do_pad=False, use_cache=False)

for i in range(15):
    sample = dataset_train[i]

    sample['event_imgs'] = F.interpolate(sample['event_imgs'].squeeze().unsqueeze(1), size=(280, 280)).squeeze().unsqueeze(-1)

    event_pixels = torch.sum(sample['event_imgs'][1:-1,...]==255, dim=[1,2,3]) + torch.sum(sample['event_imgs'][1:-1,...]==126, dim=[1,2,3])
    # gaussian smoothing
    event_pixels = torch.nn.functional.conv1d(event_pixels.view(1,1,-1).float(), torch.ones(1,1,9)/9, padding=2).squeeze()

    # find peaks in signal
    peaks = torch.zeros_like(event_pixels)
    peaks[1:-1] = (event_pixels[1:-1] > event_pixels[2:]) & (event_pixels[1:-1] > event_pixels[:-2])
    peaks[0] = (event_pixels[0] > event_pixels[1])
    peaks[-1] = (event_pixels[-1] > event_pixels[-2])

    # find vallies in signal
    valleys = torch.zeros_like(event_pixels)
    valleys[1:-1] = (event_pixels[1:-1] < event_pixels[2:]) & (event_pixels[1:-1] < event_pixels[:-2])
    valleys[0] = (event_pixels[0] < event_pixels[1])
    valleys[-1] = (event_pixels[-1] < event_pixels[-2])

    boundaries = [0] + np.where(valleys)[0].tolist() + [len(event_pixels)]
    segments = []
    for j in range(len(boundaries)-1):
        segments.append([boundaries[j], boundaries[j+1]])

    print(segments)

    counters = [0 for _ in segments]

    selected_segments = {k: False for k in range(len(segments))}

    while True:
        cur_frames = []
        for s in range(len(segments)):
            frame_id = segments[s][0] + counters[s]
            #cv2.imshow(f'segment {s}', sample['event_imgs'][frame_id].numpy())
            cur_frames.append(sample['event_imgs'][frame_id].numpy())
            counters[s] += 1
            if counters[s] > segments[s][1] - segments[s][0]:
                counters[s] = 0
            #cv2.waitKey(1)
        #cv2.imshow('all segments', np.concatenate(cur_frames, axis=1))
                
        n_rows = 3 #np.ceil(np.sqrt(len(segments)))
        n_cols = np.ceil(len(segments) / n_rows)
        # concatenate with opencv
        all_segments = np.zeros((int(n_rows*cur_frames[0].shape[0]), int(n_cols*cur_frames[0].shape[1])), np.uint8)
        for s in range(len(segments)):
            r = int(s // n_cols)
            c = int(s % n_cols)

            # add brightness to selected segments
            if selected_segments[s]:
                cur_frames[s] = np.clip(cur_frames[s] + 50, 0, 255)
                
            all_segments[r*cur_frames[0].shape[0]:(r+1)*cur_frames[0].shape[0], c*cur_frames[0].shape[1]:(c+1)*cur_frames[0].shape[1]] = cur_frames[s].squeeze()

        cv2.imshow('all segments', all_segments)
        
        # detect left click and return mouse coordinates
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                # find corresponding segment
                col_ind = int(x // cur_frames[0].shape[1])
                row_ind = int(y // cur_frames[0].shape[0])
                s = int(row_ind * n_cols + col_ind)
                print(s)
                selected_segments[s] = not selected_segments[s]
        
        cv2.setMouseCallback('all segments', mouse_callback)

        q = cv2.waitKey(50)
        if q == ord('q'):
            break
        elif q == 32:
            print(selected_segments)
            break





    

    # plt.plot(event_pixels)

    # plt.plot(peaks*event_pixels, 'o')
    # plt.plot(valleys*event_pixels, 'x')

    # plt.title(dataset_train.event_video_paths[i])

    # plt.show()