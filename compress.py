'''
    Compression using constant frame pattern on 16:9 video

    Pattern Compression because most straightfoward and removed a sizable
    amount of data.
'''

DEBUG = 1

import cv2
import os
import pickle
import io
import sys
import math

# x4 compression for ai models
DOWNSCALE_RATIO = 4

# Separates keyframes using scenedetect and every 16th frame
def decimate(video_path, output_path, sample_rate):
    # Create directory for output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initial values
    print(f'Cutting at Sample Rate: {sample_rate}')
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate times to interpolate recursively between frames
    times_to_interpolate = math.floor((total_frames / sample_rate) / 2)

    keyframes = []
    # Iterate through frames of video at sample rate
    # and save it to a compressed format for .mlpg file
    for fno in range(0, total_frames, sample_rate):

        # Collect original frame data
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, frame = vidcap.read()

        # Downscale and write encoded, downscaled image to output path
        resized = cv2.resize(frame,
            (frame.shape[1]//DOWNSCALE_RATIO, frame.shape[0]//DOWNSCALE_RATIO))
        encoded_image = cv2.imencode('.jpg', resized)
        buffer = io.BytesIO(encoded_image[1])
        keyframes.append(buffer)

        if DEBUG:
            cv2.imwrite(f'{output_path}/{fno:04d}.jpg', resized)
            print('\rWriting frame: ', fno, end='')

    # Save keyframe data
    mlpg_data = {
        'keyframes': keyframes,
        'times_to_interpolate': times_to_interpolate
    }

    # Write keyframe data to file
    with open(f'{output_path}/data.mlpg', 'wb') as f:
        pickle.dump(mlpg_data, f)

    return mlpg_data

if __name__ == '__main__':
    if sys.argv[1]:
        decimate(sys.argv[1], 'video_frames')
    else:
        decimate('video.mp4', 'video.mp4')
