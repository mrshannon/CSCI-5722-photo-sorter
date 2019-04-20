import argparse
import random
import sys
import warnings

import progressbar

from phos.common import ImageReadError, expand_image_file_list
from phos.features import FeatureExtractorID
from phos.wordlist import WordlistGenerator, save_wordlist


class CommandLineError(Exception):
    pass


class ProgressBar(progressbar.ProgressBar):

    def __init__(self, name=None, max_value=None, *args, **kwargs):
        super().__init__(*args, max_value=max_value, **kwargs)
        self._name = name

    @staticmethod
    def __bar_widgets():
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
    def __bouncing_bar_widgets():
        widgets = [
            progressbar.Timer('%(elapsed)s'), ' ',
            '(', progressbar.Counter(), ') ',
            progressbar.BouncingBar(marker='◼', left='[', right=']')
        ]
        return widgets

    def __default_widgets(self):
        if self.max_value:
            return self.__bar_widgets()
        return self.__bouncing_bar_widgets()

    def default_widgets(self):
        if self._name:
            return [f'{self._name}:  '] + self.__default_widgets()
        return self.__default_widgets()


def _progress(progress=False, name=None):
    if progress:
        # return progressbar.progressbar
        return ProgressBar(name=name)
    return lambda x: x


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


def _add_init_parser(subparsers):
    parser = subparsers.add_parser(
        'init', help='initialize an image dataset')
    parser.add_argument(
        'directory', metavar='DIR', type=str,
        help='directory to initialize the dataset in')
    parser.add_argument(
        '--wordlist', metavar='FILE', type=str, default=None,
        help='wordlist file to use, default: package provided wordlist')
    parser.add_argument(
        '--keywords', metavar='FILE', type=str, nargs='+', default=None,
        help=('keyword files and/or directories to use, '
              'default: package provided keyword files'))


def _add_index_parser(subparsers):
    parser = subparsers.add_parser(
        'index', help='index/reindex the images in the dataset')
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _add_cluster_parser(subparsers):
    parser = subparsers.add_parser(
        'cluster',
        help=('rearrange similar images into folders and duplicates into '
              'subfolders, this operation is slow'))
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


def _add_similar_parser(subparsers):
    parser = subparsers.add_parser(
        'similar', help='list images similar to the given image')
    parser.add_argument(
        'image', metavar='IMAGE', type=str,
        help='image to list similar images for')
    parser.add_argument(
        '-d', '--display', action='store_true',
        help='display images instead of listing them')


def _add_keyword_parser(subparsers):
    parser = subparsers.add_parser(
        'keyword', help='list images with the given keyword')
    parser.add_argument(
        'keyword', metavar='KEYWORD', type=str,
        help='keyword to list matching images for')
    parser.add_argument(
        '-d', '--display', action='store_true',
        help='display images instead of listing them')


def _add_keywords_parser(subparsers):
    parser = subparsers.add_parser(
        'keywords',
        help=('list available keywords and how many images in the database '
              'match each keyword or if an image is given the keywords for '
              'the image with correlations'))
    parser.add_argument(
        'image', metavar='IMAGE', type=str, nargs='?', default=None,
        help=('image to list keywords for, leave blank to list all keywords '
              'in dataset with number of images matching each keyword'))


def _add_new_keyword_parser(subparsers):
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


def _add_new_wordlist_parser(subparsers):
    parser = subparsers.add_parser(
        'new-wordlist',
        help='create a new wordlist file from a group of images')
    parser.add_argument(
        'image', metavar='IMAGE', type=str, nargs='+',
        help=('images and/or directories of images to use to generate '
              'the wordlist'))
    parser.add_argument(
        '-f', '--file', type=str, default='wordlist',
        help='name of new wordlist file, default: wordlist')
    parser.add_argument(
        '-n', '--size', metavar='N', type=int, default=1000,
        help='number of visual words to generate, default: 1000')
    parser.add_argument(
        '--max-images', metavar='N', type=int, default=None,
        help=('maximum number of files to use, if more are given the images '
              'used will be chosen at random, default: use all'))
    parser.add_argument(
        '--max-features', metavar='N', type=int, default=None,
        help=('maximum number of features to use when building the wordlist, '
              'default: use all'))
    parser.add_argument(
        '--max-features-per-image', metavar='N', type=int, default=None,
        help='maximum number of features to use per image, default: use all')
    parser.add_argument(
        '--method', type=_method_id, default=None,
        help=('set the feature extraction method to use: SURF64, SURF128, '
              'LABSURF96 (default), or LABSURF160'))
    parser.add_argument(
        '--fast', action='store_true',
        help='use fast (mini-batch) k-means intead of regular k-means')
    parser.add_argument(
        '-p', '--progress', action='store_true',
        help='show progress bar, incompatible with --verbose flag')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='enable verbose output')


def _create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'
    _add_init_parser(subparsers)
    _add_index_parser(subparsers)
    _add_cluster_parser(subparsers)
    _add_similar_parser(subparsers)
    _add_keyword_parser(subparsers)
    _add_keywords_parser(subparsers)
    _add_new_keyword_parser(subparsers)
    _add_new_wordlist_parser(subparsers)
    return parser


def _new_wordlist(args):
    # index files
    if args.progress:
        print(f"Indexing files...", file=sys.stderr, flush=True)
    images = expand_image_file_list(args.image, catch_errors=True)
    if args.max_images is not None and len(images) > args.max_images:
        images = random.sample(images, args.max_images)

    # extract features from images
    generator = WordlistGenerator(
        max_features_per_image=args.max_features_per_image,
        method_id=args.method)
    if not images:
        raise CommandLineError('no images to build wordlist from')
    for image in _progress(args.progress, 'Extracting features')(images):
        if args.verbose:
            print(image, flush=True)
        try:
            generator.add_image(image)
        except ImageReadError:
            print(f"failed to read '{image}', skipping image",
                  file=sys.stderr, flush=True)

    # generate wordlist with kmeans
    num_features = args.max_features if args.max_features \
        else generator.descriptors().shape[0]
    if args.progress:
        print(f'Using K-Means to generate wordlist from {num_features} '
              'features...',
              file=sys.stderr, flush=True)
    words = generator.generate(
        args.size, max_features=args.max_features, fast=args.fast)
    save_wordlist(args.file, words, generator.method_id)


def main():
    try:
        warnings.filterwarnings(
            "ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        args = _create_parser().parse_args()
        # print(args)
        # print('')
        try:
            if args.command == 'new-wordlist':
                _new_wordlist(args)
        except CommandLineError as err:
            print('\n' + str(err) + ', exiting', file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(2)


if __name__ == '__main__':
    main()
