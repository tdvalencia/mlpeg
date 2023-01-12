import sys, os, time, cv2
from pathlib import Path
import pandas as pd
import numpy as np
from cv2 import VideoWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//')))
from compress import decimate
from decompress import Interpolator, load_image, interpolate_recursively
os.chdir(os.path.abspath(os.path.dirname(__file__)))

input_folder = 'suite_test'
output_folder = 'suite_test_output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

direc = f'{Path.cwd()}/{input_folder}'
arr = os.listdir(direc)

interpolator = Interpolator()
times_to_interpolate = 1

df = pd.DataFrame(columns=['filename', 'mp4 sizes', 'mlpg sizes', 'ratio', 'compress time', 'decompress time'])

for file in arr:
    this_file_output = f'{output_folder}/{file[:-3]}'
    start = time.time()
    decimate(f'{input_folder}/{file}', this_file_output)
    compress_time = time.time() - start
    print(f'\nCompressing {file} took {compress_time} seconds.')

    start = time.time()
    directory = sorted(os.listdir(this_file_output))
    input_frames = []
    for image in directory:
        if image.endswith('.jpg'):
            img = load_image(f'{this_file_output}/{image}')
            input_frames.append(img)    
    print(f'Running interpolation @ times_to_interpolate={times_to_interpolate}')
    frames = list(interpolate_recursively(input_frames, times_to_interpolate, interpolator))
    decompress_time = time.time() - start
    print(f'Decompressing {file} took {decompress_time} seconds.')
    print(f'video with {len(frames)} frames')

    video = VideoWriter(f'{this_file_output}/output.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
    for image in frames:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()

    mp4_size = os.path.getsize(f'{input_folder}/{file}')
    mlpg_size = os.path.getsize(f'{this_file_output}/data.mlpg')

    df.loc[len(df.index)] = [
        file,
        mp4_size,
        mlpg_size,
        mp4_size / mlpg_size,
        compress_time,
        decompress_time
    ]
    csv = df.to_csv(f'{output_folder}/data.csv', index=False, header=True)
