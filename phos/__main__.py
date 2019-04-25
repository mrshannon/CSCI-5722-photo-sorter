import argparse
import random
import sys
import warnings

import progressbar

from phos.common import ImageReadError, ImageReadWarning, image_files
from phos.features import FeatureExtractorID, feature_name
from phos.wordlist import WordlistGenerator, save_wordlist, load_wordlist
from phos.dataset import init_dataset, find_dataset, Dataset


class CommandLineError(Exception):
    pass


class ProgressBar(progressbar.ProgressBar):

    def __init__(self, name=None, max_value=None, *args, **kwargs):
        super().__init__(*args, max_value=max_value, **kwargs)
        self._name = name

    @staticmethod
    def __known_length_widgets():
        widgets = [
            progressbar.Timer('%(elapsed)s'), ' ',
            '(', progressbar.SimpleProgress(), ') ',
            progressbar.Bar(marker='#', left='[', right=']'), ' ',
            progressbar.Percentage(), ' ',
            progressbar.ETA(
                format_not_started='ETA --:--:--',
                format_finished='            ',
                format='ETA %(eta)8s',
                format_zero='ETA 00:00:00',
                format_NA='ETA N/A')
        ]
        return widgets

    @staticmethod
    def __unknown_length_widgets():
        widgets = [
            progressbar.Timer('%(elapsed)s'), ' ',
            '(', progressbar.Counter(), ') ',
            progressbar.AnimatedMarker()
        ]
        return widgets

    def __default_widgets(self):
        if self.max_value:
            return self.__known_length_widgets()
        return self.__unknown_length_widgets()

    def default_widgets(self):
        if self._name:
            return [f'{self._name}:  '] + self.__default_widgets()
        return self.__default_widgets()


def _progress(progress=False, name=None):
    def internal(max_value=None):
        if progress:
            return ProgressBar(name=name, max_value=max_value)
        return lambda x: x
    return internal


def _print(message):
    print(message, file=sys.stderr, flush=True)


def _method_id(name):
    """Get the numeric ID of a feature extractor method by name.

    Parameters
    ----------
    name : str
        Name of the feature extractor, not case senstive.

    Returns
    -------
    FeatureExtractorID/int
        Numeric (enum) ID of the given feature extractor name.

    """
    mapping = {
        'SURF64': FeatureExtractorID.SURF64,
        'SURF128': FeatureExtractorID.SURF128,
        'LABSURF96': FeatureExtractorID.LABSURF96,
        'LABSURF160': FeatureExtractorID.LABSURF160
    }
    return mapping[name.upper()]


def _init_parser(subparsers):
    parser = subparsers.add_parser(
        'init', help='initialize an image dataset')
    parser.add_argument(
        'directory', metavar='DIR', type=str,
        help='directory to initialize the dataset in')
    parser.add_argument(
        '--method', type=_method_id, default=None,
        help=('set the feature extraction method to use: SURF64, SURF128, '
              'LABSURF96 (default), or LABSURF160'))


def _init(args):
    try:
        init_dataset(args.directory, method_id=args.method)
    except (ValueError, FileExistsError) as err:
        raise CommandLineError(str(err))


def _index_parser(subparsers):
    parser = subparsers.add_parser(
        'index', help='index/reindex the images in the dataset')
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _index(args):
    dataset = Dataset()
    with warnings.catch_warnings(record=True) as w:
        added, removed, orphaned = dataset.index_images(
            progress=_progress(args.progress, 'Indexing images'))
        if args.progress:
            message = ''
            if added:
                message += f'{len(added)} new images'
            if removed:
                message += ' (removed {len(removed)})'
            if orphaned:
                message += f', {len(orphaned)} orphaned feature files removed'
            if message:
                _print(message)
        dataset.index_features(
            progress=_progress(args.progress, 'Extracting features'))
        for warning in w:
            if warning.category is ImageReadWarning:
                _print(warning.message)
        dataset.index_words(
            progress=_progress(args.progress, 'Generating Bags of Visual Words'))


def _cluster_parser(subparsers):
    parser = subparsers.add_parser(
        'cluster',
        help=('rearrange similar images into folders and duplicates into '
              'subfolders, this operation is slow'))
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _similar_parser(subparsers):
    parser = subparsers.add_parser(
        'similar', help='list images similar to the given image')
    parser.add_argument(
        'image', metavar='IMAGE', type=str,
        help='image to list similar images for')
    parser.add_argument(
        '-d', '--display', action='store_true',
        help='display images instead of listing them')


def _keyword_parser(subparsers):
    parser = subparsers.add_parser(
        'keyword', help='list images with the given keyword')
    parser.add_argument(
        'keyword', metavar='KEYWORD', type=str,
        help='keyword to list matching images for')
    parser.add_argument(
        '-d', '--display', action='store_true',
        help='display images instead of listing them')


def _keywords_parser(subparsers):
    parser = subparsers.add_parser(
        'keywords',
        help=('list available keywords and how many images in the database '
              'match each keyword or if an image is given the keywords for '
              'the image with correlations'))
    parser.add_argument(
        'image', metavar='IMAGE', type=str, nargs='?', default=None,
        help=('image to list keywords for, leave blank to list all keywords '
              'in dataset with number of images matching each keyword'))


def _new_keyword_parser(subparsers):
    parser = subparsers.add_parser(
        'new-keyword', help='create a new keyword file from a group of images')
    parser.add_argument(
        'keyword', metavar='KEYWORD', type=str,
        help='new keyword, also the name of the keyword file')
    parser.add_argument(
        'image', metavar='IMAGE', type=str, nargs='+',
        help=('images and/or directories of images to use to generate '
              'the keyword'))
    parser.add_argument(
        '--wordlist', metavar='FILE', type=str, default=None,
        help='visual wordlist file to use, default: package provided wordlist')
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _new_wordlist_parser(subparsers):
    parser = subparsers.add_parser(
        'new-wordlist',
        help='create a new wordlist file from a group of images')
    parser.add_argument(
        'words', metavar='N', type=int, help='number of words to generate')
    parser.add_argument('file', type=str, help='name of new wordlist file')
    parser.add_argument(
        '--features', metavar='N', type=int, default=None,
        help=('maximum number of features to use when building the wordlist, '
              'default: use all'))
    parser.add_argument(
        '--kmeans', action='store_true',
        help='use regular k-means instead of the faster (mini-batch) k-means')
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _new_wordlist(args):
    if args.features and args.features < args.words:
        raise CommandLineError(
            "'--features' must be greater than the number of 'words'")
    dataset = Dataset()
    # if args.progress:
    #     print('Loading features...', file=sys.stderr, flush=True)
    try:
        generator = dataset.wordlist_generator(
            progress=_progress(args.progress, 'Loading feature files'))
    except FileNotFoundError:
        raise CommandLineError(
            "missing features files, re-run the 'index' command")
    if args.progress:
        cluster_method = 'K-Means' if args.kmeans else 'Mini Batch K-Means'
        print(f'Using {cluster_method} to generate wordlist from '
              f'{generator.num_descriptors()} features...',
              file=sys.stderr, flush=True)
    words = generator.generate(
        args.words, max_features=args.features, minibatch=(not args.kmeans))
    if args.progress:
        print('Wordlist complete!', file=sys.stderr, flush=True)
    save_wordlist(args.file, words, generator.method)


def _set_wordlist_parser(subparsers):
    parser = subparsers.add_parser(
        'set-wordlist',
        help='set the wordlist for the dataset from a file')
    parser.add_argument('file', type=str, help='wordlist file to use')
    parser.add_argument(
        '--yes', action='store_true', help="don't ask to confirm")


def _set_wordlist(args):
    if not args.yes:
        print('This will cause bags of words and keyword matches to be '
              'deleted, you will need need to re-index the dataset '
              'afterwards.')
        if input('Do you wish to proceed (yes/NO): ').lower() != 'yes':
            return
    method, words = load_wordlist(args.file)
    dataset = Dataset()
    if dataset.method != method:
        raise CommandLineError(
            f"dataset has method '{feature_name(dataset.method)}' but "
            f"wordlist was built with method '{feature_name(method)}'")
    dataset.set_wordlist(words)


def _create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'
    _init_parser(subparsers)
    _index_parser(subparsers)
    _cluster_parser(subparsers)
    _similar_parser(subparsers)
    _keyword_parser(subparsers)
    _keywords_parser(subparsers)
    _new_keyword_parser(subparsers)
    _new_wordlist_parser(subparsers)
    _set_wordlist_parser(subparsers)
    return parser


def main():
    try:
        args = _create_parser().parse_args()
        try:
            if args.command == 'init':
                _init(args)
            if args.command == 'index':
                _index(args)
            if args.command == 'new-wordlist':
                _new_wordlist(args)
            if args.command == 'set-wordlist':
                _set_wordlist(args)
        except (CommandLineError, RuntimeError) as err:
            sys.stderr.flush()
            sys.stdout.flush()
            print('\n' + str(err) + ', exiting', file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(2)


if __name__ == '__main__':
    main()
