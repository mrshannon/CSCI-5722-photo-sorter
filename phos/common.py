import mimetypes
import os
import warnings
from pathlib import Path

import rtree

import cv2 as cv
import math
import numpy as np
from PIL import Image

__all__ = ['RTree', 'Singleton', 'open_image', 'image_size',
           'cv_image', 'pil_image', 'newsize_mp', 'resize_mp', 'flatten',
           'is_image', 'is_thumbnail', 'files', 'image_files']


def RTree(dimensions, *, filename=None):
    properties = rtree.index.Property()
    properties.dimension = dimensions
    if filename:
        properties.filename = filename
        properties.storage = rtree.index.RT_Disk
        properties.pagesize = 64*dimensions
    else:
        properties.storage = rtree.index.RT_Memory
    return rtree.index.Index(properties=properties)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class ImageReadError(Exception):
    pass


class ImageReadWarning(Warning):
    pass


def open_image(file):
    try:
        return Image.open(file)
    except:
        raise ImageReadError(f"Failed to read image '{file}'")


def image_size(image):
    if isinstance(image, (str, bytes, os.PathLike)):
        with open_image(image) as img:
            return image_size(img)
    try:
        return image.shape[1], image.shape[0]
    except AttributeError:
        return image.size[1], image.size[0]


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
    if isinstance(image, (str, bytes, os.PathLike)):
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

    if isinstance(image, (str, bytes, os.PathLike)):
        return open_image(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.uint8)
    else:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 3 and bgr:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(image)


def newsize_mp(width, height, megapixels, *, upscale=False, integer=False):
    current_megapixels = (width * height) / (10 ** 6)
    div = math.sqrt(current_megapixels / megapixels)
    if integer:
        div = round(div) if div > 1 else 1 / round(1 / div)
    if div < 1 and not upscale:
        return width, height
    return int(round(width / div)), int(round(height / div))


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
    try:
        width = image.size[0]
        height = image.size[1]
    except (AttributeError, TypeError):
        width = image.shape[1]
        height = image.shape[0]

    new_shape = newsize_mp(
        width, height, megapixels, upscale=upscale, integer=integer)

    if new_shape[0] == width:
        return image

    if new_shape[0] > width:
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


def is_image(file):
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


def is_thumbnail(file):
    if not is_image(file):
        return False
    with open_image(file) as f:
        size = f.size
        return size[0] * size[1] < 16384


def files(paths, filter=None):
    filter = (lambda _: True) if filter is None else filter
    if isinstance(paths, (str, bytes, os.PathLike)):
        # single path
        for dir, _, files_ in os.walk(paths):
            for file in files_:
                fullpath = os.path.join(dir, file)
                if filter(fullpath):
                    yield Path(fullpath)
    else:
        # multiple paths
        for path in paths:
            if os.path.isdir(path):
                yield from files(path, filter)
            elif filter(path):
                yield Path(path)


def image_files(paths):
    def filter(file):
        try:
            return is_image(file) and not is_thumbnail(file)
        except ImageReadError:
            warnings.warn(
                f"failed to read '{file}', skipping image", ImageReadWarning)
            return False
    return files(paths, filter)


def get_progress(progress, max_value=None):
    return (lambda x: x) if progress is None else progress(max_value=max_value)
