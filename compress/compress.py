import cv2
from scenedetect import detect, ContentDetector
import os
import pickle
import io

DOWNSCALE_RATIO = 4

# Separates keyframes using scenedetect
def decimate(video_path, output_path, full_res=False, partial_res=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if full_res and not os.path.exists(f'{output_path}/full_res'): 
        os.makedirs(f'{output_path}/full_res')
    if partial_res and not os.path.exists(f'{output_path}/partial_res'):
        os.makedirs(f'{output_path}/partial_res')

    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    dimensions = frame.shape
    count = 0

    print('Searching for cuts...')
    scene_list = detect(video_path, ContentDetector())
    keyframe_indecies = []
    for i, scene in enumerate(scene_list):
        keyframe_indecies.append(scene[0].get_frames())
        keyframe_indecies.append(scene[1].get_frames() - 1)
        print(f'Scene {i + 1}: Start Frame: {scene[0].get_frames()}, End Frame: {scene[1].get_frames() - 1}')

    keyframes = []
    while success:
        if full_res:
            cv2.imwrite(f'{output_path}/full_res/{count:04d}.jpg', frame)

        # Downscale Images using cv2
        if partial_res:
            resized = cv2.resize(frame, 
                (dimensions[1]//DOWNSCALE_RATIO, dimensions[0]//DOWNSCALE_RATIO))
            cv2.imwrite(f'{output_path}/partial_res/{count:04d}.jpg', resized)
        if count in keyframe_indecies:
            resized = cv2.resize(frame,
                (dimensions[1]//DOWNSCALE_RATIO, dimensions[0]//DOWNSCALE_RATIO))
            encoded_image = cv2.imencode('.jpg', resized)
            buffer = io.BytesIO(encoded_image[1])
            keyframes.append(buffer)
            cv2.imwrite(f'{output_path}/{count:04d}.jpg', resized)

        success, frame = vidcap.read()
        print('\rWriting frame: ', count, end='')
        count += 1

    # Save keyframe_data
    data = {
        'dimensions': dimensions,
        'keyframe_indecies': keyframe_indecies,
        'keyframes': keyframes,
    }

    with open(f'{output_path}/data.mlpg', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    decimate('video.mp4', 'frames', full_res=True, partial_res=False)
