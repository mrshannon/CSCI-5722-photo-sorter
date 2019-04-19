import mimetypes
import os
import sys

import cv2 as cv
import math
import numpy as np
from PIL import Image

__all__ = ['cv_image', 'pil_image', 'resize_mp', 'is_image_file',
           'list_files', 'expand_file_list', 'expand_image_file_list']


class ImageReadError(Exception):
    pass


def open_image(file):
    try:
        return Image.open(file)
    except:
        raise ImageReadError(f"Failed to read image '{file}'")


def cv_image(image):
    """Convert an image into an OpenCV BGR or grayscale image.

    Parameters
    ----------
    image
        Image in one of the following forms to be converted:

            * Path to image file.
            * PIL.Image (any format).
            * 2D matrix of floats (0.0 - 1.0) or integers (0 - 255).
            * 3D matrix (RGB) of floats (0.0 - 1.0) or integers (0 - 255).

    Returns
    -------
        Either a single plane grayscale 8 bit image or a 3 plane BGR 8 bit
        image.

    """
    if isinstance(image, str):
        with open_image(image) as img:
            return cv_image(img)
    try:
        if image.mode in ('L', 'RGB'):
            image = np.array(image)
        else:
            image = np.array(image.convert('RGB'))
    except AttributeError:
        pass
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    else:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 3:
        return np.flip(image, axis=2)
    return image


def pil_image(image, bgr=True):
    """Convert to PIL image.

    Parameters
    ----------
    image
        Image in one of the following forms to be converted:

            * Path to image file.
            * PIL.Image (any format).
            * 2D matrix of floats (0.0 - 1.0) or integers (0 - 255).
            * 3D matrix (BGR or RGB if `bgr=False`) of floats (0.0 - 1.0) or
              integers (0 - 255).
    bgr
        Set to false to convert from an RGB 3 plane matrix image.

    Returns
    -------
    PIL.Image
        A PIL image from the given image data or filename.

    """

    if isinstance(image, str):
        return open_image(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    else:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 3 and bgr:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(image)


def resize_mp(image, megapixels, *, upscale=False, integer=False):
    """Resize :paramref:`image` to a given :paramref:`megapixel` size.

    Parameters
    ----------
    image
        PIL, OpenCV, or Numpy image to resize.
    megapixels : float
        Number of megapixels to resize the image to.
    upscale : bool
        Set to `True` to allow the :paramref:`image` to be upscaled, by default
        if the given :paramref:`image` is larger than :paramref:`megapixels`
        the original image will be returned un-modified.
    integer : bool
        Set to `True` to force the resize to be an integer multiple.

    Returns
    -------

    """
    width = image.shape[1]
    height = image.shape[0]
    current_megapixels = (width * height) / (10 ** 6)
    div = math.sqrt(current_megapixels / megapixels)
    if integer:
        div = round(div) if div > 1 else 1 / round(1 / div)
    new_shape = (int(round(width / div)), int(round(height / div)))

    if div == 1:
        return image

    if div < 1:
        if not upscale:
            return image
        try:
            return image.resize(new_shape, resample=Image.BILINEAR)
        except (AttributeError, TypeError):
            return cv.resize(image, new_shape, interpolation=cv.INTER_LINEAR)

    try:
        return image.resize(new_shape, resample=Image.NEAREST)
    except (AttributeError, TypeError):
        return cv.resize(image, new_shape, interpolation=cv.INTER_AREA)


def translate(dx=0, dy=0):
    """Return 2D translation matrix.

    Parameters
    ----------
    dx : float
        Amount to translate in x direction.
    dy : float
        Amount to translate in y direction.

    Returns
    -------
    3x3 array
        Translation matrix.

    """
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotate(angle):
    """Return rotation matrix.

    Parameters
    ----------
    angle : float
        Angle to rotate by.

    Returns
    -------
    3x3 array
        Rotation matrix.

    """
    return np.vstack([cv.getRotationMatrix2D((0, 0), angle, 1.0), [0, 0, 1]])


def scale(sx=1, sy=1):
    """Return scaling matrix.

    Parameters
    ----------
    sx : float
        Scale factor in x direction.
    sy : float
        Scale factor in y direction.

    Returns
    -------
    3x3 array
        Scaling matrix.

    """
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def flatten(list_of_lists):
    # https://stackoverflow.com/a/952952
    return [item for sublist in list_of_lists for item in sublist]


def is_image_file(file):
    _supported_image_mimetypes = {
        'image/bmp',
        'image/x-windows-bmp',
        'image/gif',
        'image/x-icon',
        'image/jpeg',
        'image/pjpeg',
        'image/x-portable-bitmap',
        'image/x-portable-graymap',
        'image/png',
        'image/x-portable-pixmap',
        'image/tiff',
        'image/x-tiff',
        'image/x-xbitmap',
        'image/x-xbm',
        'image/xbm',
        'image/x-xpixmap',
        'image/xpm',
    }
    return mimetypes.guess_type(file)[0] in _supported_image_mimetypes


def is_thumbnail_file(file):
    if not is_image_file(file):
        return False
    with open_image(file) as f:
        size = f.size
        return size[0] * size[1] < 16384


def list_files(dir, filter=None):
    def true(file):
        return True
    if filter is None:
        filter = true
    return [os.path.join(dir, file)
            for dir, _, files in os.walk(dir) for file in files
            if filter(os.path.join(dir, file))]


def expand_file_list(paths, filter=None, *, filter_all=False):
    def true(path):
        return True
    if filter is None:
        filter = true
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(list_files(path, filter))
        elif not filter_all or filter(path):
            files.append(path)
    return files


def expand_image_file_list(paths, *, catch_errors=False):
    def filter(file):
        if catch_errors:
            try:
                return is_image_file(file) and not is_thumbnail_file(file)
            except ImageReadError:
                print(f"failed to read '{file}', skipping image",
                      file=sys.stderr)
                return False
        return is_image_file(file) and not is_thumbnail_file(file)
    return expand_file_list(paths, filter, filter_all=True)
