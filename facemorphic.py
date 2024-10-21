import sys
print(sys.path)
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
sys.path.append("/usr/lib/python3/dist-packages/")

import os
import re
import cv2
import numpy as np
import metavision_sdk_ml
import metavision_sdk_cv
from metavision_sdk_core import EventBbox
from metavision_core.utils import get_sample
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_core.event_io import EventsIterator
from datetime import datetime, timedelta
from glob import glob
from random import shuffle
import tbr

def create_binary_frame(ev, frame_size):
    new_frame = np.zeros(frame_size, dtype=bool)
    new_frame[ev['y'], ev['x']] = 1
    return new_frame


def get_fps_stats(raw_files):
    all_fps = []
    for raw_file in raw_files:
        rgb_folder = raw_file.replace('event_', 'frames_').replace('.raw', '')
        fps = get_framerate_from_id(rgb_folder)
        all_fps.append(fps)
        if abs(fps - 30) > 1:
            print(f'Video {raw_file} has a framerate of {fps} FPS')
    print(f'Mean FPS: {np.mean(all_fps)}')
    return all_fps


def get_framerate_from_id(rgb_folder):
    rgb_frame_files = os.listdir(rgb_folder)
    rgb_frame_files = [f for f in rgb_frame_files if f.endswith('.jpg')]
    if len(rgb_frame_files) == 0:
        return 0, 0, [], []

    # sort the files
    rgb_frame_files.sort(key=lambda f: int(f.split('_')[1]))
    
    # parse timestamps from the string names. Example format: 'frame_ID_15:19:28.385380.jpg'
    timestamps = [re.search(r'(\d{2}:\d{2}:\d{2}.\d{6})', f).group(1) for f in rgb_frame_files]
    
    # convert to datetime
    timestamps = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in timestamps]
    
    # compute the time difference between frames
    frame_diff = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
    
    # convert to milliseconds
    frame_diff = [diff.total_seconds() * 1000 for diff in frame_diff]
    #print(frame_diff)
    #print(f'min: {np.min(frame_diff)}, max: {np.max(frame_diff)}')
    #print(f'mean: {np.mean(frame_diff)}, std: {np.std(frame_diff)}')
    
    total_time = (timestamps[-1] - timestamps[0]).total_seconds()
    print(f'Total time: {total_time} seconds')

    # get average fps
    fps = 1000 / np.mean(frame_diff)
    return fps, len(rgb_frame_files), timestamps, rgb_frame_files


def get_iterator(raw_file):
    # Get FPS from RGB frames
    rgb_folder = raw_file.replace('event_', 'frames_').replace('.raw', '')
    fps, num_rgb_frames, timestamps, rgb_frame_files = get_framerate_from_id(rgb_folder)
    if fps == 0:
        return None, 0, 0, [], []
    #print(fps)

    # Create iterator
    DELTA_T = int(1*1000)  # fps ms
    ev_it = EventsIterator(raw_file, start_ts=0, delta_t=DELTA_T, relative_timestamps=False)
    return ev_it, fps, num_rgb_frames, timestamps, rgb_frame_files


def get_microsecond_timestamp(date):
    return date.microsecond + date.second * 1000000 + date.minute * 1000000 * 60 + date.hour * 1000000 * 60 * 60


folder_path = '/home/becattini/Datasets/FACEMORPHIC/recordings/'
raw_files = glob(f'{folder_path}/**/*.raw', recursive=True)
print(len(raw_files))

save = True
min_diff = 9999999999999
max_diff = 0

encoding = 'TBR'
if encoding == 'TBR':
    tbr_bits = 8
    ev_width = 1280
    ev_height = 720
    tbr_incremental = True
    delta_t = 4000 # 4ms
    tbr_encoder = tbr.TemporalBinaryRepresentation(N=tbr_bits, width=ev_width, height=ev_height, incremental=tbr_incremental, cuda=True)



#shuffle(raw_files)
for n, raw_file in enumerate(raw_files):
    rgb_folder = raw_file.replace('event_', 'frames_').replace('.raw', '')
    print(raw_file)

    # create the folder if it does not exist
    save_folder = rgb_folder.replace('/frames',f'/TBR{tbr_bits}_frames')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        continue

    ev_it, fps, num_rgb_frames, timestamps, rgb_frame_files = get_iterator(raw_file)
    if fps == 0:
        continue
    ev_height, ev_width = ev_it.get_size()
    frame = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)

    # make a datetime from a timedelta of 30 milliseconds
    dt = timedelta(milliseconds=1000/fps)
    dt_datetime = datetime.min + dt
    print(dt_datetime.time())

    relative_timestamps = [t - timestamps[0] + dt_datetime for t in timestamps]  # relative timestamps + fps to consider the first frame
    next_timestamp = relative_timestamps[0]
    cur_ev = None
    frame_counter = 0
    if encoding == 'TBR':
        tbr_prev = 0
        tbr_encoder = tbr.TemporalBinaryRepresentation(N=tbr_bits, width=ev_width, height=ev_height, incremental=tbr_incremental, cuda=True)

    all_t = []
    for ev in ev_it:
        all_t.append(ev_it.get_current_time())
        if abs(ev_it.get_current_time() - get_microsecond_timestamp(next_timestamp)) > 40000:
            pass
        if cur_ev is None:
                cur_ev = ev
        elif ev_it.get_current_time() < get_microsecond_timestamp(next_timestamp):
            cur_ev = np.concatenate((cur_ev, ev))

            if encoding == 'TBR':
                # create binary slice and update incremental tbr
                if ev_it.get_current_time() - tbr_prev > delta_t:
                    tbr_prev = ev_it.get_current_time()
                    binary_frame = create_binary_frame(cur_ev, (ev_height, ev_width))
                    frame = tbr_encoder.incremental_update(binary_frame)
                    cur_ev = None
        else:
            if encoding == 'TBR':
                frame = tbr_encoder.frame.cpu().numpy()*255
            else:
                BaseFrameGenerationAlgorithm.generate_frame(cur_ev, frame)
            
            # save frame
            if save:
                cv2.imwrite(f'{save_folder}/frame_{frame_counter}.png', frame)
            else:
                #print(frame.max())
                cv2.imshow('frame', np.tile(frame[:,:,None], (1,1,3)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #cv2.imshow('frame', frame)
            #rgb_frame = cv2.imread(f'{rgb_folder}/{rgb_frame_files[frame_counter]}')
            #cv2.imshow('rgb_frame', rgb_frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            frame_counter += 1
            cur_ev = ev
            if frame_counter + 1 >= len(relative_timestamps):
                next_timestamp = datetime.max
                if abs(frame_counter + 1 - num_rgb_frames) > 0:
                    print(f'Frame {frame_counter - num_rgb_frames} is missing')
            else:
                prev_timestamp = next_timestamp
                next_timestamp = relative_timestamps[frame_counter]
                cur_diff = abs(get_microsecond_timestamp(next_timestamp) - get_microsecond_timestamp(prev_timestamp))/1000
                
                if cur_diff > max_diff:
                    max_diff = cur_diff
                if cur_diff < min_diff:
                    min_diff = cur_diff

                #print((get_microsecond_timestamp(next_timestamp) - get_microsecond_timestamp(prev_timestamp))/1000)
                #print(f'{min_diff} - {max_diff}')
    frame_counter += 1
    if encoding == 'TBR':
        frame = tbr_encoder.frame.cpu().numpy()*255
    else:
        BaseFrameGenerationAlgorithm.generate_frame(cur_ev, frame)
    # save frame
    if save:
        cv2.imwrite(f'{save_folder}/frame_{frame_counter}.png', frame)
    else:
        cv2.imshow('frame', np.tile(frame[:,:,None], (1,1,3)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print(frame_counter, num_rgb_frames, frame_counter - num_rgb_frames)
            
            

    # for frame_counter, ev in enumerate(ev_it):
    #     ts = ev_it.get_current_time()
    #     BaseFrameGenerationAlgorithm.generate_frame(ev, frame) # colors: No event (B:52, G:37, R:30); (200, 126, 64); (255, 255, 255)
    # ev_duration = ev_it.get_current_time()
    # print(ev_duration/1000000)

    # print(frame_counter - num_rgb_frames)