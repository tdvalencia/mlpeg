'''
    Tests model interpolation
'''

import sys
import mediapy as media

sys.path.insert(0, '..')

from compress import decimate
from decompress import Interpolator, interpolate_recursively, decode_keyframes

def test_interpolation(video_path):

    # splitting the video into keyframes
    video_data = decimate(video_path, 'output')

    # collecting frames
    print('Collecting frames')
    input_frames = decode_keyframes(video_data['keyframes'])

    print('Creating interpolator')
    interpolator = Interpolator()
    times_to_interpolate = 1

    print(f'Running interpolation @ times_to_interpolate = {times_to_interpolate}')
    interpolated_frames = list(interpolate_recursively(input_frames, times_to_interpolate,
                                        interpolator))
    media.write_video('output.mp4', media.to_uint8(interpolated_frames), fps=30)


if __name__ == '__main__':
    test_interpolation('../videos/saul.mp4', 480, 270)