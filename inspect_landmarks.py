import os
import cv2
from _3DMM import _3DMM
from Matrix_operations import Matrix_op
import h5py
import scipy.io as sio
import face_alignment
from tqdm import tqdm
import open3d as o3d
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

lan_paths = glob.glob('recordings/*/*/*/landmarks.npz')
show_video = None
#show_video = 'recordings/25c8ed503c504ee79218fdcee221f293/AU_27/frames_2/landmarks.npz'
show_video = 'recordings/a774e2bd46dd41f5b0e48c3a4d1277b4/AU_43/frames_2/landmarks.npz'

for lan_path in tqdm(lan_paths):
    print(lan_path)
    if show_video:
        lan_path = show_video

    landmarks = np.load(lan_path, allow_pickle=True)['arr_0'][None][0]['landmarks']

    # read frames
    frames = glob.glob(f"{'/'.join(lan_path.split('/')[:-1])}/*.jpg")
    assert len(frames) == len(landmarks)

    # natural argsort
    sorted_ids = np.argsort([int(f.split('/')[-1].split('_')[1]) for f in frames])

    frames = [frames[x] for x in sorted_ids]
    #landmarks = [landmarks[x] for x in sorted_ids]
    #np.savez_compressed(lan_path, {'landmarks': landmarks})

    if show_video:
        # save video
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('landmarks.avi', fourcc, 30, (640, 480))

        # read frames
        for n, f in enumerate(frames):
            img = cv2.imread(f)
            for l in landmarks[n]:
                cv2.circle(img, (int(l[0]), int(l[1])), 3, (0, 0, 255), -1)
            #out.write(img)
            cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        #out.release()
        break
        


