from collections.abc import Iterable
from itertools import chain
from pathlib import Path
import random
import hashlib

import numpy as np

import phos.database as db
from .common import image_size, image_files, flatten, get_progress, cv_image
from .features import create_feature_extractor
from .wordlist import WordlistGenerator

# from .image import Image

_DATASET_DIR = '.phos'

_MAX_FEATURES_PER_IMAGE = 1000


def init_dataset(path, method_id=None):
    path = Path(path).absolute()
    if path.is_dir():
        ValueError(f"path '{path}' is not a directory")
    dataset_path = path / _DATASET_DIR
    try:
        dataset_path.mkdir()
    except FileExistsError:
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


class Dataset(Iterable):

    def __init__(self, path=None):
        if path is None:
            path = find_dataset(Path.cwd())
        self._path = path
        db.connect(path / _DATASET_DIR / db._DATABASE_NAME)
        with db.session_scope() as session:
            extractor_id = int(session.query(db.KeyValue.value).filter(
                db.KeyValue.key == 'method').first()[0])
            self._feature_extractor = create_feature_extractor(extractor_id)

    @property
    def id(self):
        return self._feature_extractor.id

    def _add_images(self, images, *, progress):
        with db.session_scope() as session:
            for image in get_progress(progress)(images):
                width, height = image_size(self.absolute_path(image))
                norm_width, norm_height = (
                    self._feature_extractor.norm_size(width, height))
                session.add(db.Image(
                    path=image, width=width, height=height,
                    norm_width=norm_width, norm_height=norm_height))

    def _remove_images(self, images):
        with db.session_scope() as session:
            session.execute(
                db.Image.__table__.delete().where(db.Image.path.in_(images)))

    def index_images(self, *, search_progress=None, index_progress=None):
        fs_images = set(
            str(self.relative_path(path))
            for path in get_progress(search_progress)(image_files(self.path)))
        with db.session_scope() as session:
            db_images = set(flatten(session.query(db.Image.path).all()))
        # sorting not strictly necessary but may provide performance
        # improvements by allowing directory inode caching
        self._add_images(
            sorted(list(fs_images.difference(db_images))),
            progress=index_progress)
        self._remove_images(db_images.difference(fs_images))

    def index_features(self, *, progress=None):
        # get image id's without features
        with db.session_scope() as session:
            ids = flatten(session.query(db.Image.id).filter(
                ~db.Image.features_indexed).all())
        for id in get_progress(progress)(ids):
            # session inside loop so crashes don't undo all progress
            with db.session_scope() as session:
                image = session.query(db.Image).get(id)
                features = self._feature_extractor.extract(
                    cv_image(self.absolute_path(image.path)),
                    max_features=_MAX_FEATURES_PER_IMAGE)
                for feature in features:
                    session.add(db.Feature(
                        image=image,
                        x=feature['x'],
                        y=feature['y'],
                        angle=feature['angle'],
                        size=feature['size'],
                        descriptor=feature['descriptor'].tobytes()))
                image.features_indexed = True

    def wordlist_generator(self, *, max_features=None):
        if max_features:
            with db.session_scope() as session:
                ids = flatten(session.query(db.Feature.id).all())
                try:
                    ids = set(random.sample(ids, max_features))
                except ValueError:
                    return self.wordlist_generator()
                descriptors = session.query(db.Feature.descriptor).filter(
                    db.Feature.id.in_(ids)).all()
        else:
            with db.session_scope() as session:
                descriptors = session.query(db.Feature.descriptor).all()
        descriptors = [
            np.frombuffer(des[0], dtype=np.float32) for des in descriptors]
        return WordlistGenerator(
            descriptors, method_id=self._feature_extractor.id)

    @staticmethod
    def set_wordlist(words):
        with db.session_scope() as session:
            # remove invalid data
            session.query(db.Image).update(
                {'words_indexed': False, 'keywords_indexed': False})
            session.query(db.Word).delete()
            session.query(db.BagOfWords).delete()
            session.query(db.Keyword).delete()
            session.query(db.KeywordMatch).delete()
            # set wordlist
            wordlist_hash = hashlib.sha1(words.tobytes()).hexdigest()
            session.add(db.KeyValue(key='wordlist_hash', value=wordlist_hash))
            for word in words:
                print(word.shape)
                session.add(db.Word(descriptor=word.tobytes()))

    def __iter__(self):
        return image_files(self.path)
        # image_files = expand_image_file_list(self.path, catch_errors=True)
        # return iter()

    @property
    def path(self):
        return self._path

    def relative_path(self, path):
        return Path(path).absolute().relative_to(self.path)

    def absolute_path(self, path):
        return (self.path / Path(path)).absolute()
