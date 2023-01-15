'''
    Compression using constant frame pattern on 16:9 video

    Pattern Compression because most straightfoward and removed a sizable
    amount of data.
'''

import cv2
import os
import pickle
import io
import sys

# x4 compression for ai models
DOWNSCALE_RATIO = 4
SAMPLE_RATE = 16 # Play with this value

# Separates keyframes using scenedetect and every 16th frame
def decimate(video_path, output_path):
    # Create directory for output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initial values
    print(f'Cutting at Sample Rate: {SAMPLE_RATE}')
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    keyframe_indicies = []

    # Iterate through frames of video at sample rate
    for fno in range(0, total_frames, SAMPLE_RATE):

        # Collect original frame data
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, frame = vidcap.read()
        dimensions = frame.shape
        keyframe_indicies.append(fno)

        # Downscale and write encoded, downscaled image to output path
        resized = cv2.resize(frame,
            (dimensions[1]//DOWNSCALE_RATIO, dimensions[0]//DOWNSCALE_RATIO))
        encoded_image = cv2.imencode('.jpg', resized)
        buffer = io.BytesIO(encoded_image[1])
        keyframes.append(buffer)
        cv2.imwrite(f'{output_path}/{fno:04d}.jpg', resized)

        print('\rWriting frame: ', fno, end='')

    # Save keyframe data
    data = {
        'dimensions': dimensions,
        'keyframe_indecies': keyframe_indicies,
        'keyframes': keyframes,
    }

    # Write keyframe data to file
    with open(f'{output_path}/data.mlpg', 'wb') as f:
        pickle.dump(data, f)

    return data

if __name__ == '__main__':
    if sys.argv[1]:
        decimate(sys.argv[1], 'frames')
    else:
        decimate('video.mp4', 'frames')
