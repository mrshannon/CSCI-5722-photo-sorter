import math
import operator

import numpy as np
from sklearn.cluster import KMeans


class Clusterer:

    def __init__(self):
        self._images = []
        self._histograms = []
        self._weights = []

    def _pack(self):
        if len(self._histograms) > 2:
            self._histograms = [np.vstack(self._histograms)]

    @property
    def size(self):
        return len(self._images)

    def add_histogram(self, image, histogram, weight=1):
        self._images.append(image)
        self._histograms.append(histogram)
        self._weights.append(weight)

    def _kcluster(self, clusters):
        self._pack()
        kmeans = KMeans(n_clusters=clusters, n_jobs=-1).fit(self._histograms[0])
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        total_distance = 0
        for cluster in range(clusters):
            total_distance += (np.linalg.norm(
                self._histograms[0][labels == cluster, :] -
                centers[cluster, :], axis=1)**2).sum()
        std = math.sqrt(total_distance/(len(labels)-1))
        return centers, labels, std

    def _best_cluster(
            self, *, affinity=None, progress_printer=None):
        if affinity is None:
            affinity = 0.005
        if affinity < 0.00001:
            raise ValueError("'affinity' must be greater than 0.0001")
        old_std = 2**20  # large value
        new_std = 2**19  # less large value
        old_centers = None
        new_centers = None
        old_labels = None
        new_labels = None
        clusters = 0
        while new_std < old_std:
            # store new as old
            old_centers = new_centers
            old_labels = new_labels
            old_std = new_std
            # compute new clusters
            clusters += 1
            new_centers, new_labels, new_std = self._kcluster(clusters)
            cost = new_std*((affinity/100 + 1)**(clusters**1.5)) - new_std
            if progress_printer and (new_std + cost) < old_std:
                progress_printer(
                    f'#clusters = {clusters},  '
                    f'error = {new_std:0.3f} + {cost:0.3f}')
            new_std += cost
        return clusters-1, old_centers, old_labels

    def cluster(self, *, affinity=None, progress_printer=False):
        clusters, centers, labels = self._best_cluster(
            affinity=affinity, progress_printer=progress_printer)
        votes = {}
        for image, label, weight in zip(self._images, labels, self._weights):
            votes.setdefault(image, {}).setdefault(label, 0)
            votes[image][label] += weight
        destination_clusters = {}
        for image, labels in votes.items():
            destination_clusters[image] = max(
                labels.items(), key=operator.itemgetter(1))[0]
        return destination_clusters
