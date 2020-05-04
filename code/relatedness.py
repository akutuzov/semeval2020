import pickle
import numpy as np
from docopt import docopt
import logging
import time
from scipy.spatial.distance import pdist
from tqdm import tqdm


# Mean relatedness algorithm

def mean_relatedness_difference(word_usages1, word_usages2, metric):
    """
    Computes the difference in mean relatedness between two usage matrices.

    :param word_usages1: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param word_usages2: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param metric: a distance metric compatible with `scipy.spatial.distance.pdist`
    (e.g. 'cosine', 'euclidean')
    :return: the difference in mean relatedness between two usage matrices
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

    mean_rel1 = np.mean(pdist(usage_matrix1, metric=metric))
    mean_rel2 = np.mean(pdist(usage_matrix2, metric=metric))

    return mean_rel2 - mean_rel1


def main():
    """
    Compute difference in mean relatedness between two sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute difference in mean relatedness between two sets of contextualised 
    representations.

    Usage:
        relatedness.py [--metric=<d>, -a] <testSet> <valueFile1> <valueFile2> <outPath>

    Arguments:
        <testSet> = path to file with one target per line
        <valueFile1> = path to file containing usage matrices and snippets
        <valueFile2> = path to file containing usage matrices and snippets
        <outPath> = output path for result file

    Options:
        --metric=<d>  The distance metric, which must be compatible with 
        `scipy.spatial.distance.pdist` [default: cosine]
        -a, --abs  Store absolute (always positive) instead of raw difference

    Note:
        Assumes pickled dictionaries as input: 
        {t: (usage_matrix, snippet_list, target_pos_list) for t in targets}
        
    """)

    testset = args['<testSet>']
    valueFile1 = args['<valueFile1>']
    valueFile2 = args['<valueFile2>']
    outPath = args['<outPath>']
    distMetric = args['--metric']
    isAbsolute = args['--abs']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Load targets
    targets = []
    with open(testset, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            targets.append(target)
            # try:
            #     lemma_pos = target.split('_')
            #     lemma, pos = lemma_pos[0], lemma_pos[1]
            #     targets.append(lemma)
            # except IndexError:
            #     targets.append(target)

    # Get usages collected from corpus 1
    if valueFile1.endswith('.dict'):
        with open(valueFile1, 'rb') as f_in:
            usages1 = pickle.load(f_in)
    elif valueFile1.endswith('.npz'):
        usages1 = np.load(valueFile1)
    else:
        raise ValueError('valueFile 1: wrong format.')

    # Get usages collected from corpus 2
    if valueFile2.endswith('.dict'):
        with open(valueFile2, 'rb') as f_in:
            usages2 = pickle.load(f_in)
    elif valueFile2.endswith('.npz'):
        usages2 = np.load(valueFile2)
    else:
        raise ValueError('valueFile 2: wrong format.')

    # Print only targets to output file
    with open(outPath, 'w', encoding='utf-8') as f_out:
        for target in tqdm(targets):
            diff = mean_relatedness_difference(usages1[target], usages2[target], distMetric)
            if isAbsolute:
                diff = abs(diff)
            f_out.write('{}\t{}\n'.format(target, diff))

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
