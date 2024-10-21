import cv2
import json
from glob import glob

# read all json annotations
annotation_files = glob('recordings/*/*//*/annotation.json')

# Create a VideoWriter object
#out = cv2.VideoWriter('annotated_aus_rgb.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))

for annotation_file in annotation_files:
    print(annotation_file.split('/')[-3])
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)
        print(annotation)

        s = annotation['start']
        e = annotation['end']

        frame_path = glob(annotation_file.replace('annotation.json', '*.png'))#.replace('event_','').replace('png','jpg'))

        # natural sort
        frame_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))

        for i in range(s,e):
            img = cv2.imread(frame_path[i])
            cv2.imshow('frame', img)
            # save video
            #out.write(img)
            cv2.waitKey(5)
        #cv2.waitKey(0)

# Release the VideoWriter
#out.release()
