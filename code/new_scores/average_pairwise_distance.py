import argparse
import pickle
import numpy as np
from docopt import docopt
import logging
from scipy.spatial.distance import cdist
from tqdm import tqdm


# Average pairwise distance (APD) algorithm

def mean_pairwise_distance(word_usages1, word_usages2, metric):
    """
    Computes the mean pairwise distance between two usage matrices.

    :param word_usages1: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param word_usages2: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param metric: a distance metric compatible with `scipy.spatial.distance.cdist`
    (e.g. 'cosine', 'euclidean')
    :return: the mean pairwise distance between two usage matrices
    """
    if isinstance(word_usages1, tuple):
        usage_matrix1, _, _ = word_usages1
    else:
        usage_matrix1 = word_usages1

    if isinstance(word_usages2, tuple):
        usage_matrix2, _, _ = word_usages2
    else:
        usage_matrix2 = word_usages2

    if usage_matrix1.shape[0] == 0 or usage_matrix2.shape[0] == 0:
        raise ValueError('Zero-dimensional usage matrix.')

    return np.mean(cdist(usage_matrix1, usage_matrix2, metric=metric))


def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st npz file with the embeddings', required=True)
    arg('--input1', '-i1', help='Path to 2nd npz file with the embeddings', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    arg('--output', '-o', help='Output path (csv)', required=False)
    # arg('--metric', '-m', default='cosine',
    #     help='The distance metric, which must be compatible with `scipy.spatial.distance.cdist`')
    arg('-f', action='store_true', help='Output frequencies?')
    arg('--min_freq', type=int, default=3)
    arg('--max_samples', type=int, default=20000,
        help='Maximum number of embeddings, for time period, to use for APD calculation. '
             'If more embeddings are available, `max_samples` embeddings will be randomly sampled.')

    args = parser.parse_args()
    data_path0 = args.input0
    data_path1 = args.input1

    target_words = set([w.strip() for w in open(args.target, 'r', encoding='utf-8').readlines()])

    array0 = np.load(data_path0)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array0), data_path0))

    array1 = np.load(data_path1)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array1), data_path1))

    try:
        f_out = open(args.output, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    for target in sorted(target_words):
        frequency = np.sum([array0[target].shape[0], array1[target].shape[0]])

        if array0[target].shape[0] < args.min_freq or array1[target].shape[0] < args.min_freq:
            logger.info('{} omitted because of low frequency'.format(target))
            if args.f:
                print('\t'.join([target, '1', str(frequency)]), file=f_out)
            else:
                print('\t'.join([target, '1']), file=f_out)
            continue

        if array0[target].shape[0] > args.max_samples:
            prev = array0[target].shape[0]
            rand_indices = np.random.choice(prev, args.max_samples, replace=False)
            array0[target] = array0[target][rand_indices]
            logger.info('Choosing {} random usages from {} for {} in T0'.format(args.max_samples, prev, target))

        if array1[target].shape[0] > args.max_samples:
            prev = array1[target].shape[0]
            rand_indices = np.random.choice(prev, args.max_samples, replace=False)
            array1[target] = array1[target][rand_indices]
            logger.info('Choosing {} random usages from {} for {} in T1'.format(args.max_samples, prev, target))

        distance = mean_pairwise_distance(array0[target], array1[target], 'cosine')
        if args.f:
            print('\t'.join([target, str(distance), str(frequency)]), file=f_out)
        else:
            print('\t'.join([target, str(distance)]), file=f_out)

    if f_out:
        f_out.close()


if __name__ == '__main__':
    main()
