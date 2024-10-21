from flask import Flask, render_template, request, jsonify
import imageio
import io
import base64
import numpy as np
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
#from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM
from facemorphic_dataset import FacemorphicDataset
from tqdm import tqdm
#from torch.utils.data import DataLoader
from torchvision import transforms
import sys 
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

class Annotator:
    def __init__(self):
        
        self.params = {
            "mode": 'event_rgb',
            "max_seq_len": 999999,
            'au': 'AU'
        }
        # 'AU_1', 'AU_2', 'AU_4', 'AU_6', 'AU_7', 'AU_9', 'AU_10', 'AU_12', 'AU_14', 'AU_15', 'AU_17', 'AU_23', 'AU_24', 'AU_25', 'AU_26', 'AU_27',
        self.all_aus = [ 'AU_43',
                'AU_45', 'AU_51', 'AU_52', 'AU_53', 'AU_54', 'AU_55', 'AU_56']
        self.au_index = 0
        self.dataset = FacemorphicDataset('recordings/',
                                          split='all',
                                          mode=self.params['mode'],
                                          au_labels=[self.all_aus[self.au_index]],
                                          task=self.params['au'],
                                          toy=False,
                                          max_seq_len=self.params['max_seq_len'],
                                          clean_cache=False,
                                          do_pad=False, 
                                          use_cache=False)
        
        print('Dataset length:', len(self.dataset))
        self.index_list = np.random.permutation(len(self.dataset))
        self.cur_index = -1
    
    def get_sample(self, idx):
        if idx >= len(self.dataset):
            print('Index out of range')
            self.au_index += 1
            if self.au_index >= len(self.all_aus):
                print('All AUs annotated')
                return None
            
            print('--- NEW AU ---')
            print('AU:', self.all_aus[self.au_index])

            self.dataset = FacemorphicDataset('recordings/',
                                          split='all',
                                          mode=self.params['mode'],
                                          au_labels=[self.all_aus[self.au_index]],
                                          task=self.params['au'],
                                          toy=False,
                                          max_seq_len=self.params['max_seq_len'],
                                          clean_cache=False,
                                          do_pad=False, 
                                          use_cache=False)
            print('Dataset length:', len(self.dataset))
            self.index_list = np.random.permutation(len(self.dataset))
            self.cur_index = -1
            return None
        event_video_path = self.dataset.event_video_paths[idx]
        if os.path.isfile(f'{event_video_path}/annotation.json'):
            return None
        sample = self.dataset[idx]
        sample['event_imgs'] = F.interpolate(sample['event_imgs'].squeeze().unsqueeze(1), size=(200, 200)).squeeze().unsqueeze(-1)
        #sample['rgb_imgs'] = F.interpolate(sample['rgb_imgs'].squeeze().unsqueeze(1), size=(200, 200, 3)).squeeze()
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

        # list of segment frames
        frame_list = []
        for segment in segments:
            #frame_list.append(sample['rgb_imgs'][segment[0]:segment[1]+1, ...])
            frame_list.append(sample['event_imgs'][segment[0]:segment[1]+1, ...])
        
        return sample, segments, frame_list, event_video_path
    
    def get_next_sample(self):
        self.cur_index += 1
        print(self.cur_index, len(self.index_list))
        res = self.get_sample(self.index_list[self.cur_index])
        while res is None:
            self.cur_index += 1
            if self.cur_index >= len(self.index_list):
                # cambio au

                print('Index out of range')
                self.au_index += 1
                
                print('--- NEW AU ---')
                print('AU:', self.all_aus[self.au_index])

                self.dataset = FacemorphicDataset('recordings/',
                                            split='all',
                                            mode=self.params['mode'],
                                            au_labels=[self.all_aus[self.au_index]],
                                            task=self.params['au'],
                                            toy=False,
                                            max_seq_len=self.params['max_seq_len'],
                                            clean_cache=False,
                                            do_pad=False, 
                                            use_cache=False)
                print('Dataset length:', len(self.dataset))
                self.index_list = np.random.permutation(len(self.dataset))
                self.cur_index = -1

            res = self.get_sample(self.index_list[self.cur_index])
            print('skipping')
        sample, segments, frame_list, event_video_path = res
        return sample, segments, frame_list, event_video_path


app = Flask(__name__)
app.annotator = Annotator()


@app.route('/')
def index():
    # Assuming frames_list is a list of lists of NumPy arrays
    sample, segments, frames_list, event_video_path = app.annotator.get_next_sample()

    # Create in-memory GIFs for each set of frames
    gifs = [create_gif(frames) for frames in frames_list]

    # Convert GIFs bytes to base64 for embedding in HTML
    gifs_base64 = [base64.b64encode(gif_bytes).decode('utf-8') for gif_bytes in gifs]

    # Pass both GIFs and their indices to the template
    gifs_with_indices = list(enumerate(gifs_base64))

    max_len = min(sample['event_imgs'].shape[0], sample['rgb_imgs'].shape[0])

    whole = False
    if whole:
        event_rgb_cat = torch.concatenate((sample['event_imgs'][:max_len,...].expand(-1, -1, -1, 3), sample['rgb_imgs'][:max_len,...]), dim=2)
        whole_gif_cat = create_gif(event_rgb_cat)
        whole_gif_cat_base64 = base64.b64encode(whole_gif_cat).decode('utf-8')
    else:
        whole_gif_cat_base64 = ''

    # gif of whole video
    #whole_gif_event = create_gif(sample['event_imgs'])
    #whole_gif_event_base64 = base64.b64encode(whole_gif_event).decode('utf-8')

    #whole_gif_rgb = create_gif(sample['rgb_imgs'])
    #whole_gif_rgb_base64 = base64.b64encode(whole_gif_rgb).decode('utf-8')

    return render_template('index.html', segments=segments, gifs_with_indices=gifs_with_indices, whole_gif_cat=whole_gif_cat_base64,
                           event_video_path=event_video_path, idx=app.annotator.cur_index, num_videos=len(app.annotator.dataset.event_video_paths))

@app.route('/selected_videos', methods=['POST'])
def selected_videos():
    data = request.get_json()
    selected_videos = data.get('selectedVideos', [])
    start_annotation = data.get('start', []) 
    end_annotation = data.get('end', [])
    event_video_path = data.get('video_path', []).strip()
    print('Selected Videos:', selected_videos)
    print(start_annotation)
    print(end_annotation)

    # Process the selected video IDs
    annotation_boundaries = {'start': start_annotation, 'end': end_annotation}
    print(annotation_boundaries)

    with open(f'{event_video_path}/annotation.json', 'w') as f:
        json.dump(annotation_boundaries, f)

    return jsonify({'message': 'Selected videos received successfully'})


def create_gif(frames):
    # Squeeze the single channel dimension and convert frames to NumPy arrays
    frames_np = [torch.squeeze(frame).numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame) for frame in frames]

    # Convert frames to uint8 and normalize if needed
    frames_uint8 = [((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype(np.uint8) for frame in frames_np]

    # Create in-memory GIF
    with io.BytesIO() as gif_buffer:
        imageio.mimsave(gif_buffer, frames_uint8, format='GIF', loop=500)
        gif_bytes = gif_buffer.getvalue()

    return gif_bytes



if __name__ == '__main__':
    app.run(debug=True)
