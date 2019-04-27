from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN

from .features import FeatureExtractorID


class KeywordGenerator:

    def __init__(self):
        self._histograms = []

    def _pack(self):
        if len(self._histograms) > 2:
            self._histograms = [np.vstack(self._histograms)]

    def add_histogram(self, histogram):
        self._histograms.append(histogram)

    def generate(self, themes=None):
        self._pack()
        if themes:
            return KMeans(n_clusters=themes).fit(
                self._histograms[0]).cluster_centers_
        return MeanShift().fit(self._histograms[0]).cluster_centers_


def save_keyword(file, keyword, method):
    header = np.array(
        [method, keyword.shape[1], keyword.shape[0], 0], dtype=np.uint16)
    with open(file, 'wb') as f:
        f.write(header.tobytes())
        f.write(keyword.astype(np.float32).tobytes())


def load_keyword(file):
    keyword = Path(file).name
    with open(file, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint16, count=4)
        keyword_data = np.fromfile(
            f, dtype=np.float32).reshape(header[2], header[1])
    method = FeatureExtractorID(header[0])
    return method, keyword, keyword_data
