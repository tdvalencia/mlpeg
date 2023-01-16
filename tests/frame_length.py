import os
import cv2

file_list = os.listdir('.')

total_frames = list(set([int(cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_COUNT)) 
                for file in file_list if '.mp4' in file]))

print(total_frames)