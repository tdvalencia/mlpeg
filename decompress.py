'''
    Decompression using constant frame pattern on 16:9 video

    Image upscaling using ESPCN upscaling model and Google FILM
    for upscaling and interpolation.
'''

import sys
import pickle

import cv2, os
from cv2 import dnn_superres

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from typing import Generator, Iterable, List

# Parsing .mlpg file

def decode_keyframes(mlpg_keyframes):
  '''
    Reverses compression of frame data
      1. BytesIO
      2. cv2 imencoded

  '''
  # frames = []
  # for frame in mlpg_keyframes:
  #   buf = frame.read()
  #   nparr = np.frombuffer(buf, np.uint8)
  #   img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
  #   frames.append(img_np)

  return [cv2.cvtColor(cv2.imdecode(
    np.frombuffer(frame.read(), np.uint8), 
    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for frame in mlpg_frames]


def parse_mlpg(filename):
  '''
    Opens mlpg file and decodes keyframe data
  '''

  with open(f'{filename}', 'rb') as f:
    data = pickle.load(f)

  return data, decode_keyframes(data['keyframes'])

# Image upscaling

def upscale_file(input_path, output_path, algorithm='espcn'):
  '''
    Upscales images in a directory given a directory
  '''
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

def upscale(image_array, path_to_model_dir, algorithm='espcn'):
  '''
    Upscales an image array
  '''
  sr = dnn_superres.DnnSuperResImpl_create()
  sr.readModel(f'{path_to_model_dir}/{algorithm}_x4.pb')
  sr.setModel(algorithm, 4)

  # upscaled_image_array = []
  # for image in image_array:
  #   result = sr.upsample(image)
  #   upscaled_image_array.append(result)
  
  return [sr.upsample(image) for image in image_array]

# Google FILM frame reconstruction

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
def load_image(img_url: str):
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""
  image_data = tf.io.read_file(img_url)

  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F

"""A wrapper class for running a frame interpolation based on the FILM model on TFHub

Usage:
  interpolator = Interpolator()
  result_batch = interpolator(image_batch_0, image_batch_1, batch_dt)
  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0..1], (B,) layout.
"""

def _pad_to_align(x, align):
  """Pads image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses the Film model from TFHub
  """

  def __init__(self, align: int = 64) -> None:
    """Loads a saved model.

    Args:
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = hub.load("https://tfhub.dev/google/film/1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All inputs should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)


def interpolate_recursively(
    frames: List[np.ndarray], num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    num_recursions: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  for i in range(1, n):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    num_recursions, interpolator)
  # Separately yield the final frame.
  yield frames[-1]

if __name__ == '__main__':

  import mediapy as media

  interpolator = Interpolator()
  times_to_interpolate = 1

  if sys.argv[1]:
    mlpg_data, mlpg_frames = parse_mlpg(sys.argv[1])
  else:
    mlpg_data, mlpeg_frames = parse_mlpg('data.mlpg')

  # Run the model interpolator on the frames
  interpolated_frames = list(interpolate_recursively(
                              tf.convert_to_tensor(
                                [(frame).astype(np.float32) / 255.0
                                for frame in mlpg_frames], np.float32), 
                                times_to_interpolate, interpolator))
  
  # Upscaled all the frames
  upscaled_frames = upscale(media.to_uint8(
                            [np.array(frame) 
                            for frame in interpolated_frames]), 
                            'models')
  
  # Write the frames to a video
  media.write_video('output.mp4', upscaled_frames, fps=30)