'''
    Tests model interpolation
'''

import sys, os, cv2, time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//compress')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//decompress')))

from compress import decimate
from decompress import Interpolator, load_image, interpolate_recursively

if __name__ == '__main__':

    # splitting the video into keyframes
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    decimate('video.mp4', 'output')

    # constants
    path = '../compress/frames'
    directory = sorted(os.listdir(path))

    # collecting frames
    print('Collecting frames')
    input_frames = []
    for file in directory:
        img = load_image(f'{path}/{file}')
        input_frames.append(img)

    print('Creating interpolator')
    interpolator = Interpolator()
    times_to_interpolate = 1

    print(f'Running interpolation @ times_to_interpolate={times_to_interpolate}')
    frames = list(interpolate_recursively(input_frames, times_to_interpolate,
                                        interpolator))
    video = VideoWriter('output.avi', 0, 1, (width,height))

    print('Writing Video')
    for image in frames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
