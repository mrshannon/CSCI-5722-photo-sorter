import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from phos.features import ExtractorID, create_feature_extractor


class WordlistGenerator:

    def __init__(self, images=[], *,
                 max_features_per_image=None, method_id=None):
        self._feature_extractor = create_feature_extractor(method_id)
        self._max_features_per_file = max_features_per_image
        self._descriptors = []
        for image in images:
            self.add_image(image)

    def add_image(self, image):
        features = self._feature_extractor.extract(
            image, max_features=self._max_features_per_file)
        if np.any(np.isnan(features['descriptor'])):
            import ipdb; ipdb.set_trace()
        self._descriptors.append(features['descriptor'])

    def descriptors(self, max_features=None):
        import time
        t = time.time()
        self._pack_descriptors()
        print(f'time = {time.time() - t}')
        if (max_features is not None and
                self._descriptors[0].shape[0] > max_features):
            idx = np.random.choice(
                self._descriptors[0].shape[0],
                size=max_features, replace=False)
            return self._descriptors[0][idx, :]
        return self._descriptors[0]

    def generate(self, size, *, max_features=None, fast=False):
        if fast:
            kmeans = MiniBatchKMeans(n_clusters=size, init_size=size*100).fit(
                self.descriptors(max_features=max_features).astype(np.float64))
        else:
            kmeans = KMeans(n_clusters=size, n_jobs=-1).fit(
                self.descriptors(max_features=max_features))
        return kmeans.cluster_centers_

    def _pack_descriptors(self):
        if len(self._descriptors) > 2:
            self._descriptors = [np.vstack(self._descriptors)]
