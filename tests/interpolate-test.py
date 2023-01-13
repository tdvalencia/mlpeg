'''
    Tests model interpolation
'''

import sys, os, cv2
from cv2 import VideoWriter

sys.path.insert(0, '..')

from compress import decimate
from decompress import Interpolator, load_image, interpolate_recursively

if __name__ == '__main__':

    # splitting the video into keyframes
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    decimate('../videos/saul.mp4', 'output')

    # constants
    path = '../frames'
    directory = sorted(os.listdir(path))

    # collecting frames
    print('Collecting frames')
    # input_frames = []
    # for file in directory:
    #     if file != 'data.mlpg':
    #         img = load_image(f'{path}/{file}')
    #         input_frames.append(img)
    input_frames = [load_image(f'{path}/{file}')
                    for file in directory if file != 'data.mplg']

    print('Creating interpolator')
    interpolator = Interpolator()
    times_to_interpolate = 1

    print(f'Running interpolation @ times_to_interpolate = {times_to_interpolate}')
    frames = list(interpolate_recursively(input_frames, times_to_interpolate,
                                        interpolator))
    video = VideoWriter('output.avi', 0, 1, (480, 270))

    print('Writing Video')
    for image in frames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
