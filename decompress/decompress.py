import cv2, os
from cv2 import dnn_superres

def upscale(input_path, output_path, algorithm='espcn'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(os.path.join(os.path.dirname(__file__), f'{algorithm}_x4.pb'))
    sr.setModel(algorithm, 4)

    count = 0
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            image = cv2.imread(f'{input_path}/{filename}')
            result = sr.upsample(image)
            cv2.imwrite(f'{output_path}/{filename}', result)
            count += 1
            print('\rUpscaled frame: ', count, end='')
