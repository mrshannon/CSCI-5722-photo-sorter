from abc import abstractmethod, ABC
from enum import IntEnum

import cv2 as cv
import numpy as np

from .common import cv_image, newsize_mp, resize_mp

__all__ = ['FeatureExtractorID', 'FeatureExtractor',
           'SURFExtractor', 'LABSURFExtractor',
           'extract_keypoint', 'create_feature_extractor']


class FeatureExtractorID(IntEnum):
    """Feature extractor ID enumeration."""
    SURF64 = 1
    SURF128 = 2
    LABSURF96 = 3
    LABSURF160 = 4


_common_feature_fields = [
    ('x', np.float32),
    ('y', np.float32),
    ('angle', np.float32),
    ('size', np.uint16)]


class FeatureExtractor(ABC):

    # must define _feature_dtype in subclass

    @property
    @abstractmethod
    def id(self):
        """Extractor ID from the :class:`FeatureExtractorID` enum."""

    def norm_size(self, width, height):
        return newsize_mp(width, height, 1)

    def extract(self, image, *, max_features=None, bgr=False):
        """Extract keypoints and descriptors from given image.

        Parameters
        ----------
        image
            Any type of image that :func:`cv_image` can accept, including the
            path to an image file.
        max_features : int
            Maximum number of features to extract from the image.

        Returns
        -------
        ndarray
            Structured numpy array with the following fields:

                * x: float32
                * y: float32
                * angle: float32
                * size: uint16
                * descriptor: float32 x [96 or 160]

        """
        if not bgr:
            image = cv_image(image)
        return self._extract(resize_mp(image, 1), max_features=max_features)

    @abstractmethod
    def _extract(self, image, max_features=None):
        pass

    @staticmethod
    def _pack_results(dtype, keypoints, descriptors):
        """Pack results into a structured array of :paramref:`dtype`.

        Parameters
        ----------
        dtype
            Type descriptor of structured array to pack data into.
        keypoints : [cv.KeyPoint]
            List of OpenCV keypoints.
        descriptors : ndarray(float32 * num keypoints * N)
            An array where each row contains the descriptors for a given
            keypoint.

        Returns
        -------
        ndarray
            Structured numpy array with the following fields:

                * x: float32
                * y: float32
                * angle: float32
                * size: uint16
                * descriptor: float32 x descriptor size

        """
        results = np.empty(len(keypoints), dtype=dtype)
        results['descriptor'][:, 0:descriptors.shape[1]] = descriptors
        for result, keypoint in zip(results, keypoints):
            result['x'] = keypoint.pt[0]
            result['y'] = keypoint.pt[1]
            result['angle'] = keypoint.angle
            result['size'] = int(keypoint.size)
        return results


class SURFExtractor(FeatureExtractor):
    """Speeded-Up Robust Features (SURF) extractor.

    This is the standard SURF detector and feature extractor.  If a color image
    is given it will be converted to grayscale before processing.

    The resulting descriptor will be either 64 or 128 elements long depending
    on :paramref:`surf128`.

    Parameters
    ----------
    surf128 : bool
        Set to True to use 128 element, instead of 64 element SURF.

    """

    def __init__(self, *, surf128=False):
        super().__init__()
        self._feature_dtype = np.dtype(
            [*_common_feature_fields,
             ('descriptor', np.float32, (128 if surf128 else 64,))])
        # these are the defaults for OpenCV as of version 4.1.0
        self._detector = cv.xfeatures2d_SURF.create(
            hessianThreshold=100,
            nOctaves=4,
            nOctaveLayers=3,
            extended=surf128,
            upright=False)

    @property
    def id(self):
        if self._detector.getExtended():
            return FeatureExtractorID.SURF128
        return FeatureExtractorID.SURF64

    def _extract(self, image, max_features=None):
        """Extract keypoints and descriptors from given image.

        Parameters
        ----------
        image : ndarray(uint8)
            Grayscale (1 plane) or BGR (3 plane) image.
        max_features : int
            Maximum number of features to extract from the image.

        Returns
        -------
        ndarray
            Structured numpy array with the following fields:

                * x: float32
                * y: float32
                * angle: float32
                * size: uint16
                * descriptor: float32 x [64 or 128]

        """
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints = self._detector.detect(image)
        keypoints, descriptors = self._detector.compute(image, keypoints)
        if not keypoints:
            return np.empty((0,), dtype=self._feature_dtype)
        if max_features and max_features < len(keypoints):
            keypoints = keypoints[0:max_features]
            descriptors = descriptors[0:max_features, :]
        return self._pack_results(self._feature_dtype, keypoints, descriptors)


class LABSURFExtractor(SURFExtractor):
    """Feature extractor that adds color to the SURF descriptor.

    This feature detector works in the CIELAB color space by applying standard
    SURF to the L (lightness value) to find the keypoints and to extract either
    the first 64 or 128 elements of the descriptor.  This lightness value is
    very similar to a RGB to grayscale conversion.

    The color component is the result of subdividing the SURF keypoint
    area into a 4x4 grid and taking the average of the a* (green-red) and
    b* (blue-yellow) components from the CIELAB color space in each of the 16
    areas.  This produces an extra 32 (16 from a* and 16 from b*) descriptors.

    .. note::

        The reason the CIELAB color space was chosen is because it's components
        are normalized to human perception.  That is a unit change in L, a*, or
        b* will all be percieved roughly equally to the human eye.  It was also
        chosen for it's ability to separate intensity (lightness) and color
        (a* and b*) into orthogonal components.

    Parameters
    ----------
    color_weight : float
        Weight to apply to the color portion of the descriptor.  This can be
        used
    surf128 : bool
        Set to True to use 128 element, instead of 64 element SURF.

    """

    def __init__(self, *, color_weight=1.0, surf128=False):
        super().__init__(surf128=surf128)
        self._feature_dtype = np.dtype(
            [*_common_feature_fields,
             ('descriptor', np.float32, ((160 if surf128 else 96),))])
        self._color_weight = color_weight

    @property
    def id(self):
        if self._detector.getExtended():
            return FeatureExtractorID.LABSURF160
        return FeatureExtractorID.LABSURF96

    def _extract(self, image, max_features=None):
        """Extract keypoints and descriptors from given image.

        Parameters
        ----------
        image : ndarray(uint8)
            Grayscale (1 plane) or BGR (3 plane) image.
        max_features : int
            Maximum number of features to extract from the image.

        Returns
        -------
        ndarray
            Structured numpy array with the following fields:

                * x: float32
                * y: float32
                * angle: float32
                * size: uint16
                * descriptor: float32 x [96 or 160]

        """
        if image.ndim == 2:
            results = super()._extract(image)
            results['descriptor'][:, -32:] = 0
            return results

        lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        results = super()._extract(lab[:, :, 0], max_features=max_features)
        color = self._color_weight * self._extract_color(
            (lab[:, :, 1:3].astype(np.float32) - 128) / 127,
            results['x'], results['y'], results['angle'], results['size'])
        results['descriptor'][:, -32:] = color
        return results

    @staticmethod
    def _extract_color(ab, x, y, angle, size):
        """Extract the a* and b* color components of the descriptor.

        Parameters
        ----------
        ab : ndarray(float32)
            Two plane (a* and b*) numpy array the same width and height as the
            image array.
        x : ndarray(float32 * num keypoints)
            X coordinates of keypoints.
        y : ndarray(float32 * num keypoints)
            Y coordinates of keypoints.
        angle : ndarray(float32 * num keypoints)
            Angle of oriented keypoint.
        size : ndarray(uint16 * num keypoints)
            Size is pixels of the keypoints.

        Returns
        -------
        ndarray(float32 * num keypoints * 32)
            32 element a*b* descriptor for each keypoint.

        """
        descriptor = np.zeros((len(x), 32), dtype=np.float32)
        for i in range(len(x)):
            patch = extract_keypoint(ab, x[i], y[i], angle[i], size[i])
            patch4x4 = cv.resize(patch, (4, 4), interpolation=cv.INTER_AREA)
            descriptor[i, :] = patch4x4.ravel()
        return descriptor


def extract_keypoint(image, x, y, angle, size):
    """Extract a keypoint from an image, returning a subimage.

    Parameters
    ----------
    image : array
        Image array, number of image planes does not matter.
    x : float
        x coordinate of the keypoint to extract.
    y : float
        y coordinate of the keypoint to extract.
    angle : float
        Angle of the keypoint.
    size : int
        Size (square) of the subimage to extract.

    Returns
    -------
    ndarray
        Sub image at the keypoint.

    """
    M = np.array([[1, 0, size / 2 - x], [0, 1, size / 2 - y], [0, 0, 1]])
    M = M.dot(np.vstack(
        [cv.getRotationMatrix2D((x, y), angle, 1.0), [0, 0, 1]]))
    return cv.warpAffine(
        image, M[0:2, :], (size, size),
        flags=cv.INTER_AREA, borderMode=cv.BORDER_CONSTANT)


def create_feature_extractor(id=None):
    """Construct a feature extractor given the numeric ID/Enum.

    Parameters
    ----------
    id : FeatureExtractorID/int
        Id of the feature extractor to construct.

    Returns
    -------
    FeatureExtractor
        An instance of the corresponding feature extractor.

    """
    if id is None:
        return LABSURFExtractor()
    if id == FeatureExtractorID.SURF64:
        return SURFExtractor()
    if id == FeatureExtractorID.SURF128:
        return SURFExtractor(surf128=True)
    if id == FeatureExtractorID.LABSURF96:
        return LABSURFExtractor()
    if id == FeatureExtractorID.LABSURF160:
        return LABSURFExtractor(surf128=True)
    raise ValueError(f"unknown extractor 'id' ({id})")
