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
        return 0.

    return np.mean(cdist(usage_matrix1, usage_matrix2, metric=metric))


def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute (diachronic) distance between sets of contextualised representations.

    Usage:
        distance.py [--metric=<d> --frequency] <testSet> <valueFile1> <valueFile2> <outPath>

    Arguments:
        <testSet> = path to file with one target per line
        <valueFile1> = path to file containing usage matrices and snippets
        <valueFile2> = path to file containing usage matrices and snippets
        <outPath> = output path for result file

    Options:
        --metric=<d>  The distance metric, which must be compatible with
        `scipy.spatial.distance.cdist` [default: cosine]
        --frequency    Output frequency as well.

    Note:
        Assumes pickled dictionaries as input:
        {t: (usage_matrix, snippet_list, target_pos_list) for t in targets}

    """)

    testset = args['<testSet>']
    value_file1 = args['<valueFile1>']
    value_file2 = args['<valueFile2>']
    outpath = args['<outPath>']
    distmetric = args['--metric']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    # start_time = time.time()

    # Load targets
    targets = []
    with open(testset, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            targets.append(target)

    # Get usages collected from corpus 1
    if value_file1.endswith('.dict'):
        with open(value_file1, 'rb') as f_in:
            usages1 = pickle.load(f_in)
    elif value_file1.endswith('.npz'):
        usages1 = np.load(value_file1)
    else:
        raise ValueError('valueFile 1: wrong format.')

    # Get usages collected from corpus 2
    if value_file2.endswith('.dict'):
        with open(value_file2, 'rb') as f_in:
            usages2 = pickle.load(f_in)
    elif value_file2.endswith('.npz'):
        usages2 = np.load(value_file2)
    else:
        raise ValueError('valueFile 2: wrong format.')

    # Print only targets to output file
    with open(outpath, 'w', encoding='utf-8') as f_out:
        for target in tqdm(targets):
            frequency = np.median([usages1[target].shape[0], usages2[target].shape[0]])
            distance = mean_pairwise_distance(usages1[target], usages2[target], distmetric)
            if args['--frequency']:
                f_out.write('{}\t{}\t{}\n'.format(target, distance, frequency))
            else:
                f_out.write('{}\t{}\n'.format(target, distance))

    # logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
