import torch
from facemorphic_dataset import FacemorphicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
from torch.nn import functional as F

def _median(chunk,kernel,stride,padding):
    """Applies a median filter to a 4D tensor."""
    k = kernel
    x = F.pad(chunk, padding, mode='replicate')
    x = x.unfold(1, k[0], stride[0]).unfold(2, k[1], stride[1]).unfold(3, k[2], stride[2])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x

def compute_motion(frames):
    "basic frame difference function"
    motion = torch.zeros_like(frames)
    norm_factor = frames.shape[1]*frames.shape[2]
    for i in range(1, frames.shape[0]):
        motion[i] = torch.abs(frames[i] - frames[i-1])/norm_factor
    motion[0]=motion[1]
    
    return motion

def denoise(frames):
    "basic denoising function using a median filter trough time T x H x W x C"
    # window size 3 (frames) X 3 (height) X 3 (width) X 1 (channels)
    wnd = 5
    filtered = torch.zeros_like(frames)
    for i in range(1, frames.shape[0]-1):
        filtered[i] = _median(frames[i-1:i+2].squeeze().permute(1,2,0).unsqueeze(0), kernel=[1,1,3], stride=[1,1,3], padding=[1,1,0,0,0,0])
        # fil = Image.fromarray(filtered[i].detach().cpu().numpy().astype(np.uint8).squeeze()*255)
        # fil.save(f'filtered{i}.png')
        # fil = Image.fromarray(frames[i].detach().cpu().numpy().astype(np.uint8).squeeze()*255)
        # fil.save(f'frames{i}.png')
        # print(filtered[15].shape)
    #plt.imsave(filtered[10].detach().cpu().numpy().astype(np.uint8).squeeze(), 'filtered.png')
    return filtered

batches=2
params = {
    "mode": 'event_rgb',
    "learning_rate": 0.0001,
    "num_epochs": 10000,
    "batch_size_train": batches,
    "batch_size_test": 2,
    "max_seq_len": 50,
    "patch_size": 36,
    'au': 'AU',
    'weight_decay': 0.001
}
data_transform = None

dataset_train = FacemorphicDataset('/andromeda/datasets/FACEMORPHIC/', split='train', mode=params['mode'], 
                                   task=params['au'], toy=False, max_seq_len=params['max_seq_len'], 
                                   transform=data_transform, use_annot=True, use_cache=True)
train_dataloader = DataLoader(dataset_train, batch_size=params['batch_size_train'], shuffle=True, 
                              num_workers=1, drop_last=True)



window = 5
plot = False
# for r in range(0,16):
#     batch = train_dataloader.__iter__().__next__()
avg_diff=[]
for data in train_dataloader:
    batch,path = data
    input_rgb = batch['rgb_imgs'].float()/255.0
    input_event = batch['event_imgs'].float()/255.0
    for b in range(batches):
        filtered = input_event[b]#denoise(input_event[b])
        #filtered = F.pad(filtered, [0,0,0,0,0,0,1,0], mode='replicate')
        norm_event_factor = filtered.shape[1]*filtered.shape[2]*filtered.shape[3]
        input_event_motion = torch.sum(filtered, dim=[1,2])/norm_event_factor
        
        arg_max_event = torch.argmax(input_event_motion).item()
        
        input_event_motion = input_event_motion#/torch.max(input_event_motion)
        max_event = torch.max(input_event_motion).item()
        #print(f"Peak in event_space at frame {arg_max_event} with value {max_event}")
        input_rgb_motion = torch.sum(compute_motion(input_rgb[b]), dim=[1,2,3])
        #print(f"Input_rgb_motion tensor : {input_rgb_motion.shape}")
        wnd_left = min(arg_max_event,window)
        wnd_right = min(input_rgb_motion.shape[0]-1-arg_max_event,window)
        print(f"arg_max_event {arg_max_event} wnd_left {wnd_left} ; interval {arg_max_event-wnd_left}:{arg_max_event+wnd_right}")
        print(f"input_rgb_motion {input_rgb_motion.shape}, {input_rgb_motion[arg_max_event-wnd_left]} , {input_rgb_motion[arg_max_event+wnd_right]}")
        arg_max_rgb = arg_max_event-wnd_right + torch.argmax(input_rgb_motion[arg_max_event-wnd_left:arg_max_event+wnd_right]).item()
        input_rgb_motion = input_rgb_motion#/torch.max(input_rgb_motion)
        max_rgb = torch.max(input_rgb_motion[arg_max_event-wnd_left:arg_max_event+wnd_right]).item()
        #print(f"Peak in rgb_space at frame {arg_max_rgb}")
        avg_diff.append(arg_max_rgb-arg_max_event)
        if plot:
            plt.plot(input_event_motion.detach().cpu().numpy(), label='event')
            plt.plot(input_rgb_motion.detach().cpu().numpy(), label='rgb')
            plt.axvline(x=arg_max_event, color='b', linestyle='--')
            plt.axvline(x=arg_max_rgb, color='orange', linestyle='--',)

            plt.legend()
            plt.text(torch.argmax(input_event_motion).item()-5,torch.max(input_event_motion).item(),f"{torch.argmax(input_event_motion).item()} event")
            plt.text(arg_max_rgb,max_rgb,f"{arg_max_rgb} rgb")
            short_name = "_".join(path[b].split("/")[3:])
            plt.savefig(f"graph_{short_name}.png")
            plt.clf()
print(f"Average difference between peaks {sum(avg_diff)/len(avg_diff)}")
print(f"Max difference between peaks {max(avg_diff)}")
print(f"Sorted diffs {sorted(avg_diff)}")


