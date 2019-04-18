import argparse

from phos.features import ExtractorID


def _method_id(name):
    """Get the numeric ID of a feature extractor method by name.

    Parameters
    ----------
    name : str
        Name of the feature extractor, not case senstive.

    Returns
    -------
    ExtractorID/int
        Numeric (enum) ID of the given feature extractor name.

    """
    mapping = {
        'SURF64': ExtractorID.SURF64,
        'SURF128': ExtractorID.SURF128,
        'LABSURF96': ExtractorID.LABSURF96,
        'LABSURF160': ExtractorID.LABSURF160
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
        'keyword',  metavar='KEYWORD', type=str,
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
        '--max-features', metavar='N', type=int, default=None,
        help='maximum number of features to use per image, default: use all')
    parser.add_argument(
        '--max-files', metavar='N', type=int, default=None,
        help=('maximum number of files to use, if more are given the images '
              'used will be chosen at random, default: use all'))
    parser.add_argument(
        '--method', type=_method_id, default='LABSURF96',
        help=('set the feature extraction method to use: SURF64, SURF128, '
              'LABSURF96 (default), or LABSURF160'))
    parser.add_argument(
        '-p', '--progress', action='store_true', help='show progress bar')


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


if __name__ == '__main__':
    parser = _create_parser()
    print(parser.parse_args())
    # print('Hello')
