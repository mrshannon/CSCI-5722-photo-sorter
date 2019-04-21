import os
from pathlib import Path
from itertools import chain

from .features import create_feature_extractor
from .database import init_database, session_scope, KeyValue

_DATASET_DIR = '.phos'


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
    init_database(dataset_path)
    with session_scope() as session:
        session.add(KeyValue(key='method', value=str(int(method_id))))


def find_dataset(starting_path):
    path = Path(starting_path).absolute()
    paths = chain([path], path.parents)
    for path in paths:
        if (path / _DATASET_DIR).is_dir():
            return path / _DATASET_DIR
    raise RuntimeError(f"no '{_DATASET_DIR}' directory found")
