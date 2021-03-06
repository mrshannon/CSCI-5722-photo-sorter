import argparse
import sys
import warnings
from pathlib import Path

import progressbar

from phos.common import ImageReadWarning, files
from phos.dataset import init_dataset, Dataset
from phos.features import FeatureExtractorID, feature_name
from phos.wordlist import save_wordlist, load_wordlist
from phos.keyword import save_keyword, load_keyword


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
        if args.method is None:
            _set_wordlist(
                {'yes': True, 'file': Path(sys.prefix) / Path('phos/wordlist')})
            _set_keywords(
                {'yes': True, 'file': Path(sys.prefix) / Path('phos/keywords')})
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
                message += f' (removed {len(removed)})'
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
            progress=_progress(
                args.progress, 'Generating Bags of Visual Words'))
        dataset.index_keywords(
            progress=_progress(args.progress, 'Applying keywords to images'))


def _cluster_parser(subparsers):
    parser = subparsers.add_parser(
        'cluster',
        help=('rearrange similar images into folders and duplicates into '
              'subfolders, this operation is slow'))
    parser.add_argument(
        '-c', '--cohesion', type=float, default=2.0,
        help=('priority whole images are given over their parts, valid values '
              'are 0.0 to 100.0 with the latter only clustering based on the '
              'whole image, a value of 2.0 is considered balanced, '
              'default 2.0'))
    parser.add_argument(
        '-a', '--affinity', type=float, default=0.1,
        help=('affinity towards a single cluster, valid range is 0.01 (many '
              'clusters) to 2.0 (force a single cluster), default: 0.1'))
    parser.add_argument(
        '-g', '--global-only', action='store_true',
        help=('disable clustering based on sections of images and only use the'
              'entire scene'))
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _cluster(args):
    _index(args)
    dataset = Dataset()
    if args.progress:
        _print('Loading Bags of Visual Words...')
    clusterere = dataset.create_clusterer(
        global_only=args.global_only, image_cohesion_factor=args.cohesion)
    if clusterere.size == 0:
        raise CommandLineError(
            "missing bags of words, re-run the 'index' command")
    if args.progress:
        _print('Using K-Means to cluster images...')
    cluster_mapping = clusterere.cluster(
        affinity=args.affinity,
        progress_printer=_print if args.progress else None)
    if args.progress:
        _print('Rearranging images into clusters...')
    dataset.cluster(cluster_mapping)
    if args.progress:
        _print('Clustering complete!')


def _keyword_parser(subparsers):
    parser = subparsers.add_parser(
        'keyword', help='list images with the given keyword')
    parser.add_argument(
        'keyword', metavar='KEYWORD', type=str,
        help='keyword to list matching images for')


def _keyword(args):
    dataset = Dataset()
    try:
        images = dataset.get_images_from_keyword(args.keyword)
        for image in images:
            print(image)
    except ValueError as err:
        CommandLineError(str(err))


def _keywords_parser(subparsers):
    parser = subparsers.add_parser(
        'keywords',
        help=('list available keywords and how many images in the database '
              'match each keyword or if an image is given the keywords for '
              'the image'))
    parser.add_argument(
        'image', metavar='IMAGE', type=str, nargs='?', default=None,
        help=('image to list keywords for, leave blank to list all keywords '
              'in dataset with number of images matching each keyword'))


def _keywords(args):
    dataset = Dataset()
    if args.image:
        for keyword in dataset.get_keywords_for_image(args.image):
            print(keyword)
    else:
        for keyword, count in dataset.get_keyword_counts().items():
            print(f'{count:3d} {keyword}')


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
        '--themes', metavar='N', type=int, default=None,
        help=('set the number of themes to use, '
              'default: allow mean shift to decide'))
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _new_keyword(args):
    dataset = Dataset()
    generator = dataset.keyword_generator(args.image)
    keyword = generator.generate(themes=args.themes)
    if args.progress:
        _print(
            f'Keyword generation complete!  ({keyword.shape[0]} themes found)')
    save_keyword(args.keyword, keyword, dataset.method)


def _set_keywords_parser(subparsers):
    parser = subparsers.add_parser(
        'set-keywords',
        help='set keywords for the dataset from one or more files')
    parser.add_argument(
        'file', type=str, nargs='+', help='keyword files to use')
    parser.add_argument(
        '--yes', action='store_true', help="don't ask to confirm")


def _set_keywords(args):
    dataset = Dataset()
    # load keywords
    keywords = {}
    for file in files(args['file']):
        method, keyword, keyword_data = load_keyword(file)
        keywords[keyword] = keyword_data
        if dataset.method != method:
            raise CommandLineError(
                f"dataset has method '{feature_name(dataset.method)}' but "
                f"wordlist was built with method '{feature_name(method)}'")
    # get confirmation
    if not args['yes']:
        print('This will cause keyword matches to be deleted, you will need'
              'need to re-index the dataset afterwards.')
        if input('Do you wish to proceed (yes/NO): ').lower() != 'yes':
            return
    # set keywords
    dataset.set_keywords(keywords)


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
    method, words = load_wordlist(args['file'])
    dataset = Dataset()
    if dataset.method != method:
        raise CommandLineError(
            f"dataset has method '{feature_name(dataset.method)}' but "
            f"wordlist was built with method '{feature_name(method)}'")
    if not args['yes']:
        print('This will cause bags of words and keyword matches to be '
              'deleted, you will need need to re-index the dataset '
              'afterwards.')
        if input('Do you wish to proceed (yes/NO): ').lower() != 'yes':
            return
    dataset.set_wordlist(words)


def _create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'
    _init_parser(subparsers)
    _index_parser(subparsers)
    _cluster_parser(subparsers)
    _keyword_parser(subparsers)
    _keywords_parser(subparsers)
    _new_keyword_parser(subparsers)
    _set_keywords_parser(subparsers)
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
            if args.command == 'cluster':
                _cluster(args)
            if args.command == 'keyword':
                _keyword(args)
            if args.command == 'keywords':
                _keywords(args)
            if args.command == 'new-keyword':
                _new_keyword(args)
            if args.command == 'set-keywords':
                _set_keywords(vars(args))
            if args.command == 'new-wordlist':
                _new_wordlist(args)
            if args.command == 'set-wordlist':
                _set_wordlist(vars(args))
        except (CommandLineError, RuntimeError) as err:
            sys.stderr.flush()
            sys.stdout.flush()
            print('\n' + str(err) + ', exiting', file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(2)


if __name__ == '__main__':
    main()
