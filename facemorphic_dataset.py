import torch
import numpy as np
import os
import cv2

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from glob import glob
import json
import matplotlib.pyplot as plt


class FacemorphicDataset(Dataset):
    def __init__(self, data_dir, split, mode='event', task='AU', max_seq_len=75, central_crop=True, transform=None, toy=False, clean_cache=False, use_cache=True, do_pad=True, au_labels=None, use_annot=False):

        self.au_labels = {
            'AU': ['AU_1', 'AU_2', 'AU_4', 'AU_6', 'AU_7', 'AU_9', 'AU_10', 'AU_12',
                'AU_14', 'AU_15', 'AU_17', 'AU_23', 'AU_24', 'AU_25', 'AU_26', 'AU_27', 'AU_43',
                'AU_45', 'AU_51', 'AU_52', 'AU_53', 'AU_54', 'AU_55', 'AU_56'],
            'FREE': ['AU_FREE'],
            'READING': ['AU_READING'],
            'AU_NOHEAD': ['AU_1', 'AU_2', 'AU_4', 'AU_6', 'AU_7', 'AU_9', 'AU_10', 'AU_12',
                'AU_14', 'AU_15', 'AU_17', 'AU_23', 'AU_24', 'AU_25', 'AU_26', 'AU_27', 'AU_43',
                'AU_45'],
            'AU_HEAD': ['AU_51', 'AU_52', 'AU_53', 'AU_54', 'AU_55', 'AU_56']
        }

        self.macro_au_dict = {
            'AU_1': 0, # occhi
            'AU_2': 0,
            'AU_4': 0,
            'AU_6': 1, # occhi e bocca
            'AU_7': 0,
            'AU_9': 1, # naso (occhi e bocca)
            'AU_10': 1,
            'AU_12': 2, # bocca
            'AU_14': 2,
            'AU_15': 2,
            'AU_17': 2,
            'AU_23': 2,
            'AU_24': 2,
            'AU_25': 3, # apertura bocca
            'AU_26': 3,
            'AU_27': 3,
            'AU_43': 4, # chiusura occhi
            'AU_45': 4,
            'AU_51': 5, # testa
            'AU_52': 5,
            'AU_53': 5,
            'AU_54': 5,
            'AU_55': 5,
            'AU_56': 5
        }

        self.data_dir = data_dir
        self.central_crop=central_crop
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.split = split
        self.clean_cache = clean_cache
        self.do_pad = do_pad
        self.use_cache = use_cache
        self.use_annot = use_annot

        # Use either event, rgb or both
        assert mode in ['event', 'rgb', 'event_rgb', 'alpha', 'TBR8', 'alpha_mediapipe', 'event_mediapipe']
        self.mode = mode

        # Filter labels based on the task
        assert task in ['AU', 'FREE', 'READING', 'ALL', 'AU_NOHEAD', 'AU_HEAD']
        self.task = task
        if self.task == 'ALL':
            self.au_labels = self.au_labels['AU'] + self.au_labels['FREE'] + self.au_labels['READING']
        elif self.task == 'AU':
            self.au_labels = self.au_labels['AU']
        elif self.task == 'FREE':
            self.au_labels = self.au_labels['FREE']
        elif self.task == 'READING':
            self.au_labels = self.au_labels['READING']
        elif self.task == 'AU_NOHEAD':
            self.au_labels = self.au_labels['AU_NOHEAD']
        elif self.task == 'AU_HEAD':
            self.au_labels = self.au_labels['AU_HEAD']

        if au_labels is not None:
            self.au_labels = au_labels

        # create mapping betwenn au and index
        self.au_to_idx = {au: idx for idx, au in enumerate(self.au_labels)}
        
        self.users = sorted(os.listdir(data_dir))

        # remove user 00d38697d37c454c86599154f7de69da
        #self.users = [u for u in self.users if u != '00d38697d37c454c86599154f7de69da']
 
        self.users = self.users[::-1]

        if split == 'train':
            self.users = self.users[:int(0.8*len(self.users))]
        elif split == 'test':
            self.users = self.users[int(0.8*len(self.users)):len(self.users)]
        elif split == 'all':
            pass

        if toy:
            self.users = self.users[:2]

        self.num_users = len(self.users)

        self.rgb_video_paths = []
        self.event_video_paths = []
        self.alpha_paths = []

        alpha_path_name = 'alphas32_mediapipe'
        if mode == 'alpha':
            alpha_path_name = 'alphas32'

        # Prepare paths
        for u in tqdm(self.users):
            for au in self.au_labels:
                cur_dir = os.path.join(data_dir, u, au)
                # get_folders
                event_folders = sorted(glob(f'{cur_dir}/event_frames_*'))
                rgb_folders = sorted(glob(f'{cur_dir}/frames_*'))

                for vid_id in range(1, max(len(event_folders), len(rgb_folders)) + 1):
                    cur_event_folder = f'{cur_dir}/event_frames_{vid_id}'
                    cur_RGB_folder = f'{cur_dir}/frames_{vid_id}'

                    # Check if folders are ok
                    event_ok = os.path.exists(cur_event_folder) and len(os.listdir(cur_event_folder)) > 0
                    rgb_ok = os.path.exists(cur_RGB_folder) and len(os.listdir(cur_RGB_folder)) > 0
                    alpha_ok = os.path.exists(f'{cur_RGB_folder}/{alpha_path_name}.npz')

                    if self.use_annot:
                        annot_ok = os.path.exists(f'{cur_event_folder}/annotation.json')
                        if not annot_ok:
                            print(f'Skipping... User {u} AU {au} VID {vid_id} does not have annotation.json')
                            continue

                    if mode == 'event_rgb':
                        if not event_ok or not rgb_ok:
                            print(f'Skipping... User {u} AU {au} VID {vid_id} does not have event or rgb folders')
                            continue
                    if (mode == 'alpha' or mode == 'alpha_mediapipe') and alpha_ok:
                        self.alpha_paths.append(f'{cur_RGB_folder}/{alpha_path_name}.npz')

                    if mode == 'event_mediapipe':
                        if not event_ok or not alpha_ok:
                            print(f'Skipping... User {u} AU {au} VID {vid_id} does not have event or rgb folders')
                            continue
                    
                    if alpha_ok:
                        self.alpha_paths.append(f'{cur_RGB_folder}/{alpha_path_name}.npz')
                    if event_ok:
                        self.event_video_paths.append(cur_event_folder)
                    if rgb_ok:
                        self.rgb_video_paths.append(cur_RGB_folder)

    def __len__(self):
        if self.mode == 'event' or self.mode == 'TBR8' or self.mode == 'event_mediapipe':
            return len(self.event_video_paths)
        elif self.mode == 'rgb':
            return len(self.rgb_video_paths)
        elif self.mode == 'event_rgb':
            assert len(self.event_video_paths) == len(self.rgb_video_paths)
            return len(self.event_video_paths)
        elif self.mode == 'alpha' or self.mode == 'alpha_mediapipe':
            return len(self.alpha_paths)

    def sort_path_list(self, path_list):
        path_list.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        return path_list
    
    def pad_seq(self, seq, is_alpha=False):
        if is_alpha:
            #print(seq.shape)
            if len(seq.shape) == 1:
                seq = seq[None]
            if len(seq) < self.max_seq_len:
                seq = np.pad(seq, ((self.max_seq_len - len(seq),0),(0,0)))
        else:
            if len(seq) < self.max_seq_len:
                seq = np.pad(seq, ((self.max_seq_len - len(seq),0),(0,0),(0,0),(0,0)))
        return seq
    
    def do_central_crop(self, img):
        # Crop the central part of the image
        h, w, _ = img.shape
        if h > w:
            start = (h - w) // 2
            img = img[start:start+w, :, :]
        elif w > h:
            start = (w - h) // 2
            img = img[:, start:start+h, :]
        return img

    def read_img_seq(self, image_folder, type=None):
        img_paths = glob(f'{image_folder}/*.png')
        if len(img_paths) == 0:
            img_paths = glob(f'{image_folder}/*.jpg')

        if self.use_annot:
            cache_file = f'{image_folder}/{type}_{self.max_seq_len}_cache_with_annot.npz'
            if os.path.exists(cache_file) and self.use_cache:
                try:
                    return np.load(cache_file, allow_pickle=True)['arr_0']
                except:
                    print(cache_file)
            annot_path = f'{image_folder.replace("/frames_","/event_frames_").replace("/TBR8_frames_","/event_frames_")}/annotation.json'
            with open(annot_path,'r') as af:
                cur_annot = json.load(af)
            cur_start = cur_annot['start']
            cur_end = cur_annot['end']

            if min(cur_end, len(img_paths)) - cur_start <= 1:
                print(min(cur_end, len(img_paths)) - cur_start)
                # a small ack to avoid issues with faulty annotations. TODO: fix me
                cur_start=0
                cur_end=999999

            img_basenames = [os.path.basename(x) for x in img_paths]
            images = self.sort_path_list(img_basenames)
            img_seq = []
            images = [images[i] for i in range(cur_start, min(cur_end, len(images)))]
            for img_path in images[:self.max_seq_len]:
                img = cv2.imread(image_folder + '/' + img_path)
                if img is None:
                    print(f'Error reading {image_folder}/{img_path}')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.central_crop:
                    img = self.do_central_crop(img)
                #img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2))
                img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_NEAREST)

                img_seq.append(img)
            np_seq = np.array(img_seq)
            if type == 'event' or type == 'TBR8':
                if len(np_seq.shape) != 4:
                    print(np_seq.shape)
                np_seq = np_seq[:,:,:,1]
            np.savez_compressed(cache_file, np_seq)
        else:
            cache_file = f'{image_folder}/{self.mode}_cache.npz'
            if not self.clean_cache and os.path.exists(cache_file) and self.use_cache:
                try:
                    return np.load(cache_file, allow_pickle=True)['arr_0']
                except:
                    print(cache_file)

            img_basenames = [os.path.basename(x) for x in img_paths]
            images = self.sort_path_list(img_basenames)
            img_seq = []
            for img_path in images[:self.max_seq_len]:
                img = cv2.imread(image_folder + '/' + img_path)
                if img is None:
                    print(f'Error reading {image_folder}/{img_path}')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.central_crop:
                    img = self.do_central_crop(img)
                #img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2))
                img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_NEAREST)

                img_seq.append(img)
            np_seq = np.array(img_seq)
            #print(np_seq.shape)
            if type == 'event':
                np_seq = np_seq[:,:,:,1]
            #np.savez_compressed(cache_file, np_seq)
        return np_seq
    
    def __get_event(self, idx):
        event_path = self.event_video_paths[idx]
        if self.mode == 'TBR8':
            event_path = event_path.replace('/event_', '/TBR8_')
            event_imgs = self.read_img_seq(event_path, type='TBR8')
        else:
            event_imgs = self.read_img_seq(event_path, type='event')
        event_imgs =  event_imgs[...,None]
        label = self.au_to_idx[event_path.split('/')[-2]]
        if self.do_pad:
            event_imgs = self.pad_seq(event_imgs)

        if self.transform:
            event_imgs = torch.tensor(event_imgs)
            event_imgs = self.transform(event_imgs.permute([0,3,1,2])).permute([0,2,3,1])
        else:
            event_imgs = torch.tensor(event_imgs)
                
        sample = {'event_imgs': np.array(event_imgs), 'label': label}
        return sample

    def __get_rgb(self, idx):
            rgb_path = self.rgb_video_paths[idx]
            rgb_imgs = self.read_img_seq(rgb_path, type='rgb')
            label = self.au_to_idx[rgb_path.split('/')[-2]]
            if self.do_pad:
                    rgb_imgs = self.pad_seq(rgb_imgs)
            if self.transform:
                rgb_imgs = torch.tensor(rgb_imgs)
                rgb_imgs = self.transform(rgb_imgs.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
            sample = {'rgb_imgs': np.array(rgb_imgs), 'label': label}
            return sample
    
    def __get_alpha(self, idx):
        alpha_path = self.alpha_paths[idx]
        alpha = np.load(alpha_path, allow_pickle=True)['arr_0'][None][0]['alphas']

        # alpha is a list of elements. If one element is an empty list, then interpolate
        for i in range(len(alpha)):
            if len(alpha[i]) == 0 and i > 0:
                alpha[i] = alpha[i-1]
            elif len(alpha[i]) == 0 and i == 0:
                alpha[i] = alpha[i+1]
        alpha = np.array(alpha)

        if  alpha.shape[0] <= 1:
            deb = 1

        label = self.au_to_idx[alpha_path.split('/')[-3]]

        # cut with annotation
        if self.use_annot:
            annot_path = alpha_path.replace("/frames_","/event_frames_").replace('alphas32.npz','annotation.json').replace('alphas32_mediapipe.npz','annotation.json')
            with open(annot_path,'r') as af:
                cur_annot = json.load(af)
            cur_start = cur_annot['start']
            cur_end = min(cur_annot['end'], len(alpha))
            alpha = alpha[cur_start:cur_end]
            alpha = alpha[:self.max_seq_len]

        if self.do_pad:
            alpha = self.pad_seq(alpha, is_alpha=True)
        sample = {'alpha': np.array(alpha), 'label': label}
        return sample

    def __getitem__(self, idx):
        if self.mode == 'event' or self.mode == 'TBR8':
            sample = self.__get_event(idx)
        elif self.mode == 'rgb':
            sample = self.__get_rgb(idx)
        elif self.mode == 'event_rgb':
            sample = self.__get_event(idx)
            sample.update(self.__get_rgb(idx))
        elif self.mode == 'alpha' or self.mode == 'alpha_mediapipe':
            sample = self.__get_alpha(idx)
            if np.array(sample['alpha']).dtype != np.float64:
                print(f'---------------------------{idx}-------------------{self.alpha_paths[idx]}')
        elif self.mode == 'event_mediapipe':
            sample = self.__get_event(idx)
            sample.update(self.__get_alpha(idx))
        sample = {k: torch.from_numpy(np.array(v)) for k,v in sample.items()}
        return sample


if False:


    data_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    
    ds = FacemorphicDataset('recordings', split='train', mode='TBR8', task='AU', toy=False, max_seq_len=50, transform=data_transform, use_annot=True, use_cache=False)

    print(len(ds))

    for sample in tqdm(ds):
        pass
    #sample = ds[0]

    # Inspect outliers
    video_lengths = [len(os.listdir(x)) for x in ds.event_video_paths]
    plt.plot(sorted(video_lengths),'.')
    plt.show()

    ids_long_videos = np.argsort(video_lengths)
    for i in ids_long_videos[-10:]:
        print(ds.event_video_paths[i], video_lengths[i])

    pass