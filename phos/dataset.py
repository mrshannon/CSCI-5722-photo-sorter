import hashlib
from functools import partial
from itertools import chain
from pathlib import Path
import operator

import numpy as np

import phos.database as db
from .cluster import Clusterer
from .common import RTree, files, image_files, flatten, get_progress, cv_image
from .features import (feature_name, feature_descriptor_size,
                       read_features_header, load_features,
                       create_feature_extractor, feature_rtree)
from .wordlist import WordlistGenerator
from .keyword import KeywordGenerator

_DATASET_DIR = '.phos'

_MAX_FEATURES_PER_IMAGE = 1000

_FEATURE_EXTENSION = '.fea'


def _make_hidden(path):
    path = Path(path)
    return path.parent / ('.' + str(path.name))


def init_dataset(path, method_id=None):
    path = Path(path).absolute()
    if path.is_dir():
        ValueError(f"path '{path}' is not a directory")
    dataset_path = path / _DATASET_DIR
    dataset_path.mkdir(exist_ok=True)
    if (dataset_path / Path(db._DATABASE_NAME)).is_file():
        raise FileExistsError(f"existing dataset at '{dataset_path}'")
    method_id = create_feature_extractor(method_id).id
    db.init(dataset_path)
    with db.session_scope() as session:
        session.add(db.KeyValue(key='method', value=str(int(method_id))))


def find_dataset(starting_path):
    path = Path(starting_path).absolute()
    paths = chain([path], path.parents)
    for path in paths:
        if ((path / _DATASET_DIR).is_dir() and
                (path / _DATASET_DIR / db._DATABASE_NAME)):
            return path
    raise RuntimeError(f"no '{_DATASET_DIR}' directory found")


class Dataset:

    def __init__(self, path=None):
        if path is None:
            path = find_dataset(Path.cwd())
        self._path = path
        db.connect(path / _DATASET_DIR / db._DATABASE_NAME)
        with db.session_scope() as session:
            extractor_id = int(session.query(db.KeyValue.value).filter(
                db.KeyValue.key == 'method').first()[0])
            self._feature_extractor = create_feature_extractor(extractor_id)

    def _index_image(self, session, path):
        feature_path = self._get_feature_file(path)
        has_features = (
                feature_path.is_file() and
                read_features_header(feature_path)[0] == self.method)
        relative_path = str(self.relative_path(path))
        image = session.query(db.Image).filter(
            db.Image.path == relative_path).one_or_none()
        if image:
            image.has_features = has_features
            return False
        session.add(db.Image(path=relative_path, has_features=has_features))
        return True

    @staticmethod
    def _remove_missing_images(session, fs_images):
        fs_images = set(str(path) for path in fs_images)
        db_images = set(flatten(session.query(db.Image.path).all()))
        removed = db_images.difference(fs_images)
        for image in session.query(db.Image).filter(db.Image.path.in_(removed)):
            session.delete(image)
        return list(removed)

    def _remove_orphaned_features(self, fs_images):
        def filter_(path):
            return str(path).endswith(_FEATURE_EXTENSION)

        orphans = []
        feature_files = set(self._get_feature_file(path) for path in fs_images)
        for file in files(self.path, filter=filter_):
            if ((self.absolute_path(file) not in feature_files) or
                    read_features_header(file)[0] != self.method):
                try:
                    file.relative_to(Path.cwd())
                    file.unlink()
                except ValueError:
                    raise RuntimeError(
                        f"cannot delete '{file}', file outside of current "
                        "directory")
                    pass
                orphans.append(self.relative_path(file))
        return orphans

    def _get_feature_file(self, image_path):
        return _make_hidden(
            self.absolute_path(image_path).with_suffix(_FEATURE_EXTENSION))

    def _get_feature_files(self):
        with db.session_scope() as session:
            return [self._get_feature_file(path[0])
                    for path in session.query(db.Image.path).all()]

    def _get_wordlist_rtree(self):
        words = self.get_wordlist()
        wordtree = RTree(feature_descriptor_size(self.method))
        for id, descriptor in words.items():
            descriptor_ = np.frombuffer(descriptor, dtype=np.float32)
            wordtree.insert(id, np.tile(descriptor_, 2))
        return wordtree

    def _classify_descriptors(self, descriptors, *, wordlist=None):
        if wordlist is None:
            wordlist = self._get_wordlist_rtree()
        return np.array([list(wordlist.nearest(np.tile(des, 2), 1))[0]
                         for des in descriptors], dtype=np.uint16)

    @staticmethod
    def _word_histogram(num_words, feature_words, feature_tree,
                        width, height, divisions, i, j):
        if i >= divisions or j >= divisions:
            raise ValueError("Both 'i' and 'j' must be less than 'divisions'.")
        dx = width / divisions
        dy = height / divisions
        idx = list(feature_tree.intersection(
            (dx * i, dy * j, dx * (i + 1), dy * (j + 1))))
        return np.bincount(feature_words[idx], minlength=num_words)

    @staticmethod
    def _insert_bag_of_words(session, hist_fun, image, divisions, row, column):
        hist = hist_fun(divisions, row, column).astype(np.float32)
        if hist.max() > 0:
            hist = hist / hist.max()
            session.add(db.BagOfWords(
                image=image, divisions=divisions, row=row, column=column,
                word_histogram=hist.tobytes()))

    def _create_bag_of_words_for_image(self, image_id, size, wordlist):
        with db.session_scope() as session:
            image = session.query(db.Image).get(image_id)
            method, width, height, features = load_features(
                self._get_feature_file(image.path))
            if method != self.method:
                raise RuntimeError(
                    f"dataset uses '{feature_name(self.method)}' method but "
                    f"features were extracted with '{feature_name(method)}' "
                    "method")
            words = self._classify_descriptors(
                features['descriptor'], wordlist=wordlist)
            feature_tree = feature_rtree(features)
            word_histogram = partial(
                self._word_histogram, size, words, feature_tree, width, height)
            # total image
            self._insert_bag_of_words(session, word_histogram, image, 1, 0, 0)
            # rule of 3rds
            for i in range(3):
                for j in range(3):
                    self._insert_bag_of_words(
                        session, word_histogram, image, 3, i, j)
            # rule of 5ths
            for i in range(5):
                for j in range(5):
                    self._insert_bag_of_words(
                        session, word_histogram, image, 5, i, j)
            image.has_words = True

    def _move_image_into(self, image_id, dest_dir):
        dest_dir = self.absolute_path(dest_dir)
        with db.session_scope() as session:
            image = session.query(db.Image).get(image_id)
            image_path = self.absolute_path(image.path)
            dest_dir.mkdir(parents=True, exist_ok=True)
            image_path.rename(dest_dir / image_path.name)
            self._get_feature_file(image_path).rename(
                self._get_feature_file(dest_dir / image_path.name))
            image.path = str(self.relative_path(dest_dir / image_path.name))

    def index_images(self, *, progress=None):
        added = []
        fs_images = []
        with db.session_scope() as session:
            for path in get_progress(progress)(image_files(self.path)):
                relative_path = self.relative_path(path)
                if self._index_image(session, path):
                    added.append(relative_path)
                fs_images.append(relative_path)
            removed = self._remove_missing_images(session, fs_images)
        orphaned = self._remove_orphaned_features(fs_images)
        return added, removed, orphaned

    def index_features(self, *, progress=None):
        new_features = []
        with db.session_scope() as session:
            images = session.query(db.Image).filter(
                ~db.Image.has_features).all()
            for image in get_progress(progress)(images):
                path = self.absolute_path(image.path)
                image_data = cv_image(path)
                features = self._feature_extractor.extract(
                    image_data, max_features=_MAX_FEATURES_PER_IMAGE)
                self._feature_extractor.save_features(
                    self._get_feature_file(path), image_data, features)
                image.has_features = True
                new_features.append(image.path)
        return new_features

    def index_words(self, *, progress=None):
        with db.session_scope() as session:
            image_ids = flatten(session.query(db.Image.id).filter(
                ~db.Image.has_words).all())
        num_words = len(self.get_wordlist())
        if num_words == 0:
            raise RuntimeError('no wordlist set for dataset')
        wordlist = self._get_wordlist_rtree()
        for id in get_progress(progress)(image_ids):
            self._create_bag_of_words_for_image(id, num_words, wordlist)

    @staticmethod
    def get_keywords(id=False):
        keywords = {}
        with db.session_scope() as session:
            for keyword in session.query(db.Keyword):
                themes = [np.frombuffer(theme.word_histogram, dtype=np.float32)
                          for theme in keyword.themes]
                if id:
                    keywords[keyword.id] = np.vstack(themes)
                else:
                    keywords[keyword.name] = np.vstack(themes)
        return keywords

    def _get_keywords_rtree(self):
        keywords = self.get_keywords(id=True)
        histogram_length = next(iter(keywords.values())).shape[1]
        keywords_tree = RTree(histogram_length)
        for id, histograms in keywords.items():
            for histogram in histograms:
                keywords_tree.insert(id, np.tile(histogram, 2))
        return keywords_tree

    def _index_image_keywords(self, image_id, keywords_tree):
        with db.session_scope() as session:
            image = session.query(db.Image).get(image_id)
            image_keyword_ids = {}
            for bag_of_words in image.words:
                divisions = bag_of_words.divisions
                histogram = np.frombuffer(
                    bag_of_words.word_histogram, dtype=np.float32)
                for id in keywords_tree.nearest(np.tile(histogram, 2), 1):
                    image_keyword_ids.setdefault(id, 0)
                    image_keyword_ids[id] += 1/(divisions**1.8)
            image_keyword_ids = {
                k: v for k, v in image_keyword_ids.items() if v >= 1}
            for keyword_id in set(image_keyword_ids):
                keyword = session.query(db.Keyword).get(keyword_id)
                session.add(db.KeywordMatch(image=image, keyword=keyword))
            image.has_keywords = True

    def index_keywords(self, *, progress=None):
        with db.session_scope() as session:
            image_ids = flatten(session.query(db.Image.id).filter(
                ~db.Image.has_keywords).all())
        keywords = self.get_keywords(id=True)
        if len(keywords) == 0:
            return False
        keywords_tree = self._get_keywords_rtree()
        for id in get_progress(progress)(image_ids):
            self._index_image_keywords(id, keywords_tree)

    @staticmethod
    def get_images_from_keyword(keyword):
        images = []
        with db.session_scope() as session:
            keyword = session.query(db.Keyword).filter(
                db.Keyword.name == keyword).one_or_none()
            if keyword is None:
                return ValueError(
                    f"keyword '{keyword}' is not in this dataset")
            for match in keyword.images:
                images.append(match.image.path)
        return images

    @staticmethod
    def get_keyword_counts():
        keywords = {}
        with db.session_scope() as session:
            for keyword in session.query(db.Keyword):
                keywords[keyword.name] = len(keyword.images)
        return {k: v for (k,v) in
                sorted(keywords.items(),
                       key=operator.itemgetter(1), reverse=True)}

    def get_keywords_for_image(self, image):
        image = self.relative_path(image)
        with db.session_scope() as session:
            image = session.query(db.Image).filter(
                db.Image.path == str(image)).one_or_none()
            if image:
                return set([match.keyword.name for match in image.keywords])
            return set()

    @staticmethod
    def create_clusterer(*, global_only=False, image_cohesion_factor=2):
        if image_cohesion_factor < 0:
            raise ValueError("'image_cohesion_factor' cannot be less than 0")
        clusterere = Clusterer()
        with db.session_scope() as session:
            for bags_of_words in session.query(db.BagOfWords):
                if global_only and bags_of_words.divisions > 1:
                    continue
                histogram = np.frombuffer(
                    bags_of_words.word_histogram, dtype=np.float32)
                weight = 1 / (bags_of_words.divisions ** image_cohesion_factor)
                clusterere.add_histogram(
                    bags_of_words.image_id,
                    histogram, weight=weight)
        return clusterere

    def cluster(self, cluster_mapping=None):
        if cluster_mapping is None:
            cluster_mapping = self.create_clusterer().cluster()
        for image_id, cluster in cluster_mapping.items():
            self._move_image_into(
                image_id, Path('clusters') / Path(str(cluster)))
        # clean up empty cluster directories
        if self.absolute_path('clusters').is_dir():
            for cluster_dir in self.absolute_path('clusters').glob('*'):
                if (cluster_dir.is_dir() and
                        len(list(cluster_dir.glob('*'))) == 0):
                    cluster_dir.rmdir()

    def keyword_generator(self, paths):
        images = set(str(self.relative_path(file)) for file in image_files(paths))
        generator = KeywordGenerator()
        with db.session_scope() as session:
            query = session.query(db.BagOfWords.word_histogram).\
                join(db.BagOfWords.image).\
                filter(db.Image.path.in_(images)).\
                filter(db.BagOfWords.divisions == 1)
            for word_histogram in query:
                generator.add_histogram(
                    np.frombuffer(word_histogram[0], dtype=np.float32))
        return generator

    def set_keywords(self, keywords):
        with db.session_scope() as session:
            # remove invalid data
            session.query(db.Image).update({'has_keywords': False})
            session.query(db.Keyword).delete()
            session.query(db.KeywordTheme).delete()
            session.query(db.KeywordMatch).delete()
            # add keywords
            for keyword, data in keywords.items():
                keyword_ = db.Keyword(name=keyword)
                session.add(keyword_)
                for histogram in data:
                    session.add(db.KeywordTheme(
                        keyword=keyword_, word_histogram=histogram.tobytes()))

    @staticmethod
    def get_wordlist():
        words = {}
        with db.session_scope() as session:
            for word in session.query(db.Word):
                descriptor = np.frombuffer(word.descriptor, dtype=np.float32)
                words[word.id - 1] = descriptor
        return words

    def wordlist_generator(self, *, progress=None):
        feature_files = self._get_feature_files()
        # check for the existence of files
        for file in feature_files:
            if not file.is_file():
                raise FileNotFoundError(
                    f"feature file '{str(file)}' does not exist")
        # load feature files
        generator = WordlistGenerator(method_id=self.method)
        for file in get_progress(progress)(feature_files):
            method, _, _, features = load_features(file)
            if method == self.method:
                generator.add_descriptors(features['descriptor'])
        return generator

    @staticmethod
    def set_wordlist(words):
        with db.session_scope() as session:
            # remove invalid data
            session.query(db.Image).update(
                {'has_words': False, 'has_keywords': False})
            session.query(db.Word).delete()
            session.query(db.BagOfWords).delete()
            session.query(db.Keyword).delete()
            session.query(db.KeywordMatch).delete()
            # set wordlist
            wordlist_hash = hashlib.sha1(words.tobytes()).hexdigest()
            session.add(db.KeyValue(key='wordlist_hash', value=wordlist_hash))
            for id, word in enumerate(words):
                session.add(db.Word(id=(id + 1), descriptor=word.tobytes()))

    def relative_path(self, path):
        return Path(path).absolute().relative_to(self.path)

    def absolute_path(self, path):
        return (self.path / Path(path)).absolute()

    @property
    def path(self):
        return self._path

    @property
    def method(self):
        return self._feature_extractor.id
