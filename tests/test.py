import sys, os, time, math
from pathlib import Path
import pandas as pd
import mediapy as media
import numpy as np

sys.path.insert(0, '..')

from compress import decimate
from decompress import Interpolator, interpolate_recursively, upscale, decode_keyframes

input_folder = 'suite_test'

direc = f'{Path.cwd()}/{input_folder}'
file_list = os.listdir(direc)

SAMPLE_RATE = [2, 4, 8, 16, 32, 64, 128]
interpolator = Interpolator()

df = pd.DataFrame(columns=['filename', 'mp4 sizes', 'mlpg sizes', 'ratio (mlpg/mp4)', 'compress time', 'decompress time', 'sample rate'])

for sample_rate in SAMPLE_RATE:

    output_folder = f'suite_test_output/{sample_rate}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in file_list:

        # Test and store compression output
        this_file_output = f'{output_folder}/{file[:-3]}'
        start = time.time()

        mlpg_data = decimate(f'{input_folder}/{file}', this_file_output, sample_rate)

        compress_time = time.time() - start
        print(f'\nCompressing {file} took {compress_time} seconds.')

        times_to_interpolate = int(math.log2(sample_rate));

        # Test and store decompression output
        start = time.time()

        mlpg_frames = decode_keyframes(mlpg_data['keyframes'])
        normalized_float32_mlpg_frames = [(frame).astype(np.float32) / 255.0
                                        for frame in mlpg_frames]
        print(f'Running interpolation @ times_to_interpolate = {times_to_interpolate}')
        interpolated_frames = list(interpolate_recursively(normalized_float32_mlpg_frames, 
                                    times_to_interpolate, interpolator))

        interpolated_numpy_frames = [np.array(frame)
                                for frame in interpolated_frames]

        upscaled_frames = upscale(media.to_uint8(interpolated_numpy_frames), '../models/')

        media.write_video(f'{this_file_output}/{file}.mp4', upscaled_frames, fps=60)

        decompress_time = time.time() - start
        print(f'Decompressing {file} took {decompress_time} seconds.')
        print(f'video with {len(interpolated_numpy_frames)} frames')

        mp4_size = os.path.getsize(f'{input_folder}/{file}')
        mlpg_size = os.path.getsize(f'{this_file_output}/data.mlpg')

        df.loc[len(df.index)] = [
            file,
            mp4_size,
            mlpg_size,
            mlpg_size / mp4_size,
            compress_time,
            decompress_time,
            sample_rate
        ]
        csv = df.to_csv(f'{output_folder}/suite_test.csv', index=False, header=True)
