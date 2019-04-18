import math
import cv2 as cv
import numpy as np
from PIL import Image

__all__ = ['cv_image', 'pil_image', 'resize_mp']


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
        with Image.open(image) as img:
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
        return Image.open(image)
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
    current_megapixels = (width * height)/(10**6)
    div = math.sqrt(current_megapixels/megapixels)
    if integer:
        div = round(div) if div > 1 else 1/round(1/div)
    new_shape = (int(round(width/div)), int(round(height/div)))

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
