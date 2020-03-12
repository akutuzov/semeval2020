# python3
# coding: utf-8

import logging
import numpy as np
from docopt import docopt
from scipy.stats import entropy
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def cluster(usage_matrix, time_labels, algorithm, args_dict, word, max_examples=5000):
    """
    :param word: target word
    :param usage_matrix: a matrix of contextualised word representations of shape
    (num_usages, model_dim)
    :param time_labels: an array of 0s and 1s labelling each word representation in the usage matrix
    :param algorithm:the clustering algorithm: DBSCAN or Affinity Propagation
    :param args_dict: the sklearn parameters of the chosen clustering algorithm
    :param max_examples: maximum number of usages to cluster (if higher, will be downsampled)
    :return: n_clusters - the number of clusters in the given usage matrix
             labels - a list of labels, one for each contextualised representation
    """
    if algorithm.lower() not in ['dbscan', 'db', 'affinity', 'affinity propagation', 'ap']:
        raise ValueError('Invalid clustering method:', algorithm)

    if usage_matrix.shape[0] > max_examples:
        prev = usage_matrix.shape[0]
        rand_indices = np.random.choice(prev, max_examples, replace=False)
        usage_matrix = usage_matrix[rand_indices]
        time_labels = time_labels[rand_indices]
        logger.info('Choosing {} random rows from {} for {}'.format(max_examples, prev, word))
    usage_matrix = StandardScaler().fit_transform(usage_matrix)

    if algorithm.lower() in ['dbscan', 'db']:
        clustering = DBSCAN(**args_dict).fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        clustering = AffinityPropagation(**args_dict).fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels))

    assert len(labels) == usage_matrix.shape[0] == len(time_labels)
    logger.info('{} has {} clusters in {} instances'.format(word, n_clusters, len(labels)))
    return n_clusters, labels, time_labels


def compute_jsd_scores(filepath1, filepath2, algorithm, args_dict, words, threshold=0.8, rat=0.05):
    """
    :param threshold: ratio of time bin occurrences in a cluster to consider it novel
    :param rat: ratio of all occurrences in a cluster to consider it at all
    :param filepath1: path to .npz file containing a dictionary
    :param filepath2: path to .npz file containing a dictionary
    {lemma: usage matrix for lemma in targets}
    :param algorithm: the clustering algorithm: DBSCAN or Affinity Propagation
    :param args_dict: the sklearn parameters of the chosen clustering algorithm
    :param words: set of target words
    :return: jsd_scores - a dictionary {lemma: jsd score for lemma in words}
    """
    usage_dict1 = np.load(filepath1)
    usage_dict2 = np.load(filepath2)

    sense_distributions1 = {}
    sense_distributions2 = {}

    shifts = {l: {} for l in set(words)}

    for word in words:
        if word in usage_dict1 and word in usage_dict2:
            usage_matrix = np.vstack((usage_dict1[word], usage_dict2[word]))
            time_labels = np.ones(usage_matrix.shape[0])
            time_labels[usage_dict1[word].shape[0]:] = 2

            if usage_matrix.shape[0] > 0:
                num_senses_w, labels, time_labels = cluster(usage_matrix, time_labels, algorithm,
                                                            args_dict, word)

                sense_distributions1[word] = np.zeros(num_senses_w)
                sense_distributions2[word] = np.zeros(num_senses_w)

                # Count frequency of each sense in both corpora
                sense_label_ids = sorted(set(labels))
                for sense_label_id in sense_label_ids:
                    for cl_label, t_label in zip(labels, time_labels):
                        if t_label == 1:
                            sense_distributions1[word][sense_label_id] += (
                                    sense_label_id == cl_label)
                        else:
                            sense_distributions2[word][sense_label_id] += (
                                    sense_label_id == cl_label)

                # Normalise to obtain sense (probability) distribution
                for sense_label_id in range(num_senses_w):
                    if sum(sense_distributions1[word]) > 0:
                        sense_distributions1[word][sense_label_id] /= sum(
                            sense_distributions1[word])
                    if sum(sense_distributions2[word]) > 0:
                        sense_distributions2[word][sense_label_id] /= sum(
                            sense_distributions2[word])

                senses = {l: [] for l in set(labels)}
                assert num_senses_w == len(senses)

                # Count frequency of each time bin in each sense
                for label, instance in zip(labels, time_labels):
                    senses[label].append(instance)

                for sense in senses:
                    if sense == -1:
                        continue
                    total = len(senses[sense])
                    if total / len(time_labels) < rat:
                        continue
                    ones = senses[sense].count(1) / total
                    twos = senses[sense].count(2) / total
                    if ones > threshold:
                        shifts[word][sense] = {'old': (round(ones, 3), total)}
                    elif twos > threshold:
                        shifts[word][sense] = {'new': (round(twos, 3), total)}
            else:
                print('No vectors for', word)
        else:
            print(word, 'not found in npz dictionary.')

    jsd_scores = {}
    for word in words:
        try:
            jsd_scores[word] = {'jsd': jsd(sense_distributions1[word], sense_distributions2[word]),
                                'shift': 0}
        except KeyError:
            jsd_scores[word] = {'jsd': 1.0, 'shift': 1}
        logger.info('======')
        logger.info(word)
        logger.info(round(jsd_scores[word]['jsd'], 3))
        if shifts[word]:
            logger.info(shifts[word])
            jsd_scores[word]['shift'] = 1
    return jsd_scores


def main():
    """
    Get number of senses given two diachronic sets of usage representations.
    """

    # Get the arguments
    args = docopt("""Get number of senses for two sets of usage representations.

    Usage:
        jsd.py <targets> <distributionsFile1> <distributionsFile2> <ratio> <outPath>

    Arguments:
        <targets> = path to target words file
        <distributionsFile1> = path to .npz file containing a dictionary that maps words to usage matrices (corpus1)
        <distributionsFile2> = path to .npz file containing a dictionary that maps words to usage matrices (corpus2)
        <ratio> = ratio of word occurrences in a cluster to consider it at all
        <outPath> = output filepath *without extension* for csv file with a JSD value for each target word
                    (format: 'lemma jsd')
    """)

    filepath1 = args['<distributionsFile1>']
    filepath2 = args['<distributionsFile2>']
    ratio = float(args['<ratio>'])
    outpath = args['<outPath>']

    target_words = list(set([w.strip() for w in open(args['<targets>'], 'r').readlines()]))

    clustering_method = 'AP'
    args_dicts = {
        'DB': {
            'eps': 0.2,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'leaf_size': 10,
            'p': None,
        },
        'AP': {
            'damping': 0.5,
            'max_iter': 200,
            'convergence_iter': 15,
            'preference': None,
            'affinity': 'euclidean'
        }
    }
    logger.info('Clustering using %s and sense frequency ratio %f' % (clustering_method, ratio))

    jsd_scores = compute_jsd_scores(
        filepath1, filepath2, clustering_method, args_dicts[clustering_method], target_words, rat=ratio)
    with open('{}'.format(outpath), 'w', encoding='utf-8') as f:
        for word, score in jsd_scores.items():
            # f.write("{}\t{}\t{}\n".format(word, score['jsd'], score['shift']))
            f.write("{}\t{}\n".format(word, score['shift']))


if __name__ == '__main__':
    main()
