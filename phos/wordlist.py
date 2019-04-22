import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from phos.features import FeatureExtractorID, create_feature_extractor

__all__ = ['WordlistGenerator', 'save_wordlist', 'load_wordlist']


class WordlistGenerator:

    def __init__(self, images=[], *,
                 max_features_per_image=None, method_id=None):
        self._method_id = method_id
        self._feature_extractor = create_feature_extractor(method_id)
        self._max_features_per_file = max_features_per_image
        self._descriptors = []
        for image in images:
            self.add_image(image)

    @property
    def method_id(self):
        return self._feature_extractor.id

    def add_image(self, image):
        features = self._feature_extractor.extract(
            image, max_features=self._max_features_per_file)
        self._descriptors.append(features['descriptor'])

    def descriptors(self, max_features=None):
        self._pack_descriptors()
        if (max_features is not None and
                self._descriptors[0].shape[0] > max_features):
            idx = np.random.choice(
                self._descriptors[0].shape[0],
                size=max_features, replace=False)
            return self._descriptors[0][idx, :]
        return self._descriptors[0]

    def generate(self, size, *, max_features=None, minibatch=True):
        if minibatch:
            kmeans = MiniBatchKMeans(n_clusters=size, init_size=size*100).fit(
                self.descriptors(max_features=max_features).astype(np.float64))
        else:
            kmeans = KMeans(n_clusters=size, n_jobs=-1).fit(
                self.descriptors(max_features=max_features))
        return kmeans.cluster_centers_

    def _pack_descriptors(self):
        if len(self._descriptors) > 2:
            self._descriptors = [np.vstack(self._descriptors)]


def save_wordlist(file, words, method):
    # import ipdb; ipdb.set_trace()
    header = np.array(
        [method, words.shape[1], words.shape[0], 0], dtype=np.uint16)
    with open(file, 'wb') as f:
        f.write(header.tobytes())
        f.write(words.astype(np.float32).tobytes())


def load_wordlist(file):
    with open(file, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint16, count=4)
        method = FeatureExtractorID(header[0])
        words = np.fromfile(f, dtype=np.float32).reshape(header[2], header[1])
        return method, words
