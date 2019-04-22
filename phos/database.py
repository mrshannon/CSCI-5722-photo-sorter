from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import (Column, ForeignKey, Integer, Float, String, Unicode,
                        LargeBinary, create_engine)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, backref

from .common import Singleton

__all__ = ['KeyValue', 'Image', 'Feature', 'Word', 'BagOfWords',
           'Keyword', 'KeywordMatch', 'Database', 'init_database', 'connect']

_DATABASE_NAME = 'dataset.sql'

_Base = declarative_base()


class KeyValue(_Base):
    __tablename__ = 'key_value'
    id = Column(Integer, primary_key=True)
    key = Column(String(64), nullable=False, index=True)
    value = Column(Unicode(256))


class Image(_Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True)
    path = Column(Unicode(256), nullable=False, index=True)
    width = Column(Integer)
    height = Column(Integer)


class Feature(_Base):
    __tablename__ = 'feature'
    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    angle = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=False)
    image = relationship(
        'Image', backref=backref('features', cascade='all, delete-orphan'))
    descriptor = Column(LargeBinary())


class Word(_Base):
    __tablename__ = 'word'
    id = Column(Integer, primary_key=True)
    descriptor = Column(LargeBinary())


class BagOfWords(_Base):
    __tablename__ = 'bag_of_words'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=False)
    image = relationship(
        'Image', backref=backref('words', cascade='all, delete-orphan'))
    layer = Column(Integer, nullable=False)
    row = Column(Integer, nullable=False)
    column = Column(Integer, nullable=False)
    word_histogram = Column(LargeBinary(), nullable=False)


class Keyword(_Base):
    __tablename__ = 'keyword'
    id = Column(Integer, primary_key=True)
    name = Column(Unicode(256), nullable=False, index=True)
    word_histogram = Column(LargeBinary(), nullable=False)


class KeywordMatch(_Base):
    __tablename__ = 'keyword_match'
    id = Column(Integer, primary_key=True)
    match = Column(Float, nullable=False)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=False)
    image = relationship('Image', backref='keywords')
    keyword_id = Column(Integer, ForeignKey('keyword.id'), nullable=False)
    keyword = relationship(
        'Keyword', backref=backref('images', cascade='all, delete-orphan'))


class Database(metaclass=Singleton):
    _path: str = ''
    _engine: Engine = None

    def __init__(self, path=None):
        if path is None:
            return
        path = Path(path).absolute()
        if path.is_dir():
            path = path / Path(_DATABASE_NAME)
        path = 'sqlite:///' + str(path)
        if self._path != path:
            self._path = path
            self._engine = create_engine(self._path)
            self._session = sessionmaker(bind=self._engine)
        self._init_check()

    def _init_check(self):
        if self._engine is None:
            raise ValueError(f"'path' never provided")

    @property
    def path(self):
        return self._path

    @property
    def engine(self):
        return self._engine

    @property
    def session(self):
        return self._session


def init_database(path):
    db = Database(path)
    _Base.metadata.create_all(db.engine)


def connect(path):
    Database(path)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Database().session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
