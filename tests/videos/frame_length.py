'''
    Standardize test video total frame count to 300
    of Inter4k dataset
'''

import os
import cv2

file_list = os.listdir('.')
dict = {}

for file in file_list:
    if '.mp4' in file:
        dict[file] = cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_COUNT)

cwd = os.getcwd()
for (file, frames) in dict.items():
    if frames < 300:
        print(file, frames)
        os.remove(file)
