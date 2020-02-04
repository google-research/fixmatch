# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various ops for augmentation."""

import math

import tensorflow as tf
import tensorflow_addons as tfa


# Default replace value
REPLACE_VALUE = 128


def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.

  A value of factor 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor.
    image2: An image Tensor.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor.
  """
  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)
  return tf.saturate_cast(image1 + factor * (image2 - image1), tf.uint8)


def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image):
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, image_shape[2] - 1], 1)

  replace = tf.constant([REPLACE_VALUE, REPLACE_VALUE, REPLACE_VALUE, 1],
                        image.dtype)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(
      image,
      [0, 0, 0],
      [image_shape[0], image_shape[1], image_shape[2] - 1])
  return image


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  threshold = tf.saturate_cast(threshold, image.dtype)
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128
  threshold = tf.saturate_cast(threshold, image.dtype)
  added_im = tf.cast(image, tf.int32) + tf.cast(addition, tf.int32)
  added_im = tf.saturate_cast(added_im, tf.uint8)
  return tf.where(image < threshold, added_im, image)


def invert(image):
  """Inverts the image pixels."""
  return 255 - tf.convert_to_tensor(image)


def invert_blend(image, factor):
  """Implements blend of invert with original image."""
  return blend(invert(image), image, factor)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  grayscale_im = tf.image.rgb_to_grayscale(image)
  mean = tf.reduce_mean(tf.cast(grayscale_im, tf.float32))
  mean = tf.saturate_cast(mean + 0.5, tf.uint8)

  degenerate = tf.ones_like(grayscale_im, dtype=tf.uint8) * mean
  degenerate = tf.image.grayscale_to_rgb(degenerate)

  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = tf.cast(8 - bits, image.dtype)
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees):
  """Equivalent of PIL Rotation."""
  # Convert from degrees to radians
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = tfa.image.transform_ops.rotate(wrap(image), radians)
  return unwrap(image)


def translate_x(image, pixels):
  """Equivalent of PIL Translate in X dimension."""
  image = tfa.image.translate_ops.translate(wrap(image), [-pixels, 0])
  return unwrap(image)


def translate_y(image, pixels):
  """Equivalent of PIL Translate in Y dimension."""
  image = tfa.image.translate_ops.translate(wrap(image), [0, -pixels])
  return unwrap(image)


def shear_x(image, level):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1]
  image = tfa.image.transform_ops.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image)


def shear_y(image, level):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1]
  image = tfa.image.transform_ops.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops."""

  def scale_channel(channel):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(channel), tf.float32)
    hi = tf.cast(tf.reduce_max(channel), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      return tf.saturate_cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(channel), lambda: channel)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def autocontrast_blend(image, factor):
  """Implements blend of autocontrast with original image."""
  return blend(autocontrast(image), image, factor)


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_im = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.squeeze(tf.saturate_cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_im)

  # Blend the final result
  return blend(result, orig_im, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def equalize_blend(image, factor):
  """Implements blend of equalize with original image."""
  return blend(equalize(image), image, factor)


def _convolve_image_with_kernel(image, kernel):
  num_channels = tf.shape(image)[-1]
  kernel = tf.tile(kernel, [1, 1, num_channels, 1])
  image = tf.expand_dims(image, axis=0)
  convolved_im = tf.nn.depthwise_conv2d(
      tf.cast(image, tf.float32), kernel, strides=[1, 1, 1, 1], padding='SAME')
  # adding 0.5 for future rounding, same as in PIL:
  # https://github.com/python-pillow/Pillow/blob/555e305a60d7fcefd1ad4aa6c8fd879e2f474192/src/libImaging/Filter.c#L101  # pylint: disable=line-too-long
  convolved_im = convolved_im + 0.5
  return tf.squeeze(convolved_im, axis=0)


def blur(image, factor):
  """Blur with the same kernel as ImageFilter.BLUR."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class BLUR(BuiltinFilter):
  #     name = "Blur"
  #     # fmt: off
  #     filterargs = (5, 5), 16, 0, (
  #         1, 1, 1, 1, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 1, 1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  blur_kernel = tf.constant([[1., 1., 1., 1., 1.],
                             [1., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 1.],
                             [1., 1., 1., 1., 1.]],
                            dtype=tf.float32,
                            shape=[5, 5, 1, 1]) / 16.0
  blurred_im = _convolve_image_with_kernel(image, blur_kernel)
  return blend(image, blurred_im, factor)


def smooth(image, factor):
  """Smooth with the same kernel as ImageFilter.SMOOTH."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class SMOOTH(BuiltinFilter):
  #     name = "Smooth"
  #     # fmt: off
  #     filterargs = (3, 3), 13, 0, (
  #         1, 1, 1,
  #         1, 5, 1,
  #         1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  smooth_kernel = tf.constant([[1., 1., 1.],
                               [1., 5., 1.],
                               [1., 1., 1.]],
                              dtype=tf.float32,
                              shape=[3, 3, 1, 1]) / 13.0
  smoothed_im = _convolve_image_with_kernel(image, smooth_kernel)
  return blend(image, smoothed_im, factor)


def rescale(image, level):
  """Rescales image and enlarged cornet."""
  # TODO(kurakin): should we do center crop instead?
  # TODO(kurakin): add support of other resize methods
  # See tf.image.ResizeMethod for full list
  size = image.shape[:2]
  scale = level * 0.25
  scale_height = tf.cast(scale * size[0], tf.int32)
  scale_width = tf.cast(scale * size[1], tf.int32)
  cropped_image = tf.image.crop_to_bounding_box(
      image,
      offset_height=scale_height,
      offset_width=scale_width,
      target_height=size[0] - scale_height,
      target_width=size[1] - scale_width)
  rescaled = tf.image.resize(cropped_image, size, tf.image.ResizeMethod.BICUBIC)
  return tf.saturate_cast(rescaled, tf.uint8)


NAME_TO_FUNC = {
    'Identity': tf.identity,
    'AutoContrast': autocontrast,
    'AutoContrastBlend': autocontrast_blend,
    'Equalize': equalize,
    'EqualizeBlend': equalize_blend,
    'Invert': invert,
    'InvertBlend': invert_blend,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Blur': blur,
    'Smooth': smooth,
    'Rescale': rescale,
}
