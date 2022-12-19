# import subprocess

# youtube_dl_path = "youtube-dl.exe"
# subprocess.call([youtube_dl_path, "-o", "video.mp4", "https://www.youtube.com/watch?v=Maxio3hiE80"])

import sys, os, cv2, time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//compress')))
from compress import decimate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//decompress')))
from decompress import upscale

os.chdir(os.path.abspath(os.path.dirname(__file__)))
decimate('video.mp4', 'output', full_res=True, partial_res=True)

def percent_diff(folder1, folder2):
    total = 0
    filenames1 = os.listdir(folder1)
    filenames2 = os.listdir(folder2)
    for x in range(len(filenames1)):
        img1 = cv2.imread(f'{folder1}/{filenames1[x]}', 0)
        img2 = cv2.imread(f'{folder2}/{filenames2[x]}', 0)
        res = cv2.absdiff(img1, img2)
        res = res.astype(np.uint8)
        total = total + (np.count_nonzero(res) * 100)/ res.size
    print('Average percent difference: ', total/len(filenames1))

print('\nUpscale Algorithm: ESPCN')
start_time = time.time()
upscale('output/partial_res', 'output/espcn', algorithm='espcn')
print("\n--- %s seconds ---" % (time.time() - start_time))
percent_diff('output/full_res', 'output/espcn')

print('Upscale Algorithm: FSRCNN')
start_time = time.time()
upscale('output/partial_res', 'output/fsrcnn', algorithm='fsrcnn')
print("\n--- %s seconds ---" % (time.time() - start_time))
percent_diff('output/full_res', 'output/fsrcnn')

print('Upscale Algorithm: EDSR')
start_time = time.time()
upscale('output/partial_res', 'output/edsr', algorithm='edsr')
print("\n--- %s seconds ---" % (time.time() - start_time))
percent_diff('output/full_res', 'output/edsr')

