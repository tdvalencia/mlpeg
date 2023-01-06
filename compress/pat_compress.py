'''
    Compression using constant frame pattern on 16:9 video
'''

import cv2
from scenedetect import detect, ContentDetector
import os
import pickle
import io

# Why x4 compression?: For algoritm
DOWNSCALE_RATIO = 4
SAMPLE_RATE = 16 # Play with this value

# Separates keyframes using scenedetect and every 16th frame
def decimate(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vidcap = cv2.VideoCapture(video_path)

    keyframes = []
    keyframe_indicies = []
    print(f'Cutting at Sample Rate: {SAMPLE_RATE}')
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fno in range(0, total_frames, SAMPLE_RATE):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, frame = vidcap.read()
        dimensions = frame.shape
        keyframe_indicies.append(fno)

        resized = cv2.resize(frame,
            (dimensions[1]//DOWNSCALE_RATIO, dimensions[0]//DOWNSCALE_RATIO))
        encoded_image = cv2.imencode('.jpg', resized)
        buffer = io.BytesIO(encoded_image[1])
        keyframes.append(buffer)
        cv2.imwrite(f'{output_path}/{fno:04d}.jpg', resized)

        print('\rWriting frame: ', fno, end='')

    # Save keyframe_data
    data = {
        'dimensions': dimensions,
        'keyframe_indecies': keyframe_indicies,
        'keyframes': keyframes,
    }

    with open(f'{output_path}/data.mlpg', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    decimate('video.mp4', 'frames')
