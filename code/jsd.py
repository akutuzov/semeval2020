# python3
# coding: utf-8

from docopt import docopt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AffinityPropagation
import logging
import numpy as np
from scipy.spatial import distance

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def cluster(usage_matrix, algorithm, args_dict, word, max_examples=10000):
    """
    :param word: target word
    :param usage_matrix: a matrix of contextualised word representations of shape
    (num_usages, model_dim)
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
        logger.info('Choosing {} random rows from {} for {}'.format(max_examples, prev, word))
    # logger.info('{} matrix shape: {}'.format(word, usage_matrix.shape))
    usage_matrix = StandardScaler().fit_transform(usage_matrix)

    if algorithm.lower() in ['dbscan', 'db']:
        clustering = DBSCAN(**args_dict).fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        clustering = AffinityPropagation(**args_dict).fit(usage_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels))

    assert len(labels) == usage_matrix.shape[0]
    logger.info('{} has {} clusters'.format(word, n_clusters))
    return n_clusters, labels


def compute_jsd_scores(filepath1, filepath2, algorithm, args_dict, words):
    """
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

    for word in words:
        if word in usage_dict1 and word in usage_dict2:
            n_usages_corpus1 = usage_dict1[word].shape[0]
            usage_matrix = np.vstack((usage_dict1[word], usage_dict2[word]))

            if usage_matrix.shape[0] > 0:
                num_senses_w, labels = cluster(usage_matrix, algorithm, args_dict, word)

                sense_distributions1[word] = np.zeros(num_senses_w)
                sense_distributions2[word] = np.zeros(num_senses_w)

                # Count frequency of each sense in both corpora
                for sense_label_id in range(num_senses_w):
                    sense_distributions1[word][sense_label_id] = sum(sense_label_id == labels[:n_usages_corpus1])
                    sense_distributions2[word][sense_label_id] = sum(sense_label_id == labels[n_usages_corpus1:])

                # Normalise to obtain sense (probability) distribution
                for sense_label_id in range(num_senses_w):
                    sense_distributions1[word][sense_label_id] /= sum(sense_distributions1[word])
                    sense_distributions2[word][sense_label_id] /= sum(sense_distributions2[word])
            else:
                print('no vectors for ', word)
        else:
            print(word, 'not found in npz dictionary.')

    jsd_scores = {}
    for word in words:
        jsd_scores[word] = jsd(sense_distributions1[word], sense_distributions2[word])
        # jsd_scores[word] = distance.jensenshannon(sense_distributions1[word], sense_distributions2[word])

    return jsd_scores


def main():
    """
    Get number of senses given two diachronic sets of usage representations.
    """

    # Get the arguments
    args = docopt("""Get number of senses for two sets of usage representations.

    Usage:
        jsd.py <targets> <distributionsFile1> <distributionsFile2> <outPath>

    Arguments:
        <targets> = path to target words file
        <distributionsFile1> = path to .csv file containing word sense distributions
        <distributionsFile2> = path to .csv file containing word sense distributions
        <outPath> = output filepath *without extension* for csv file with a JSD value for each target word
                    (format: 'lemma jsd')
    """)

    filepath1 = args['<distributionsFile1>']
    filepath2 = args['<distributionsFile2>']
    outPath = args['<outPath>']

    target_words = list(set([w.strip() for w in open(args['<targets>'], 'r').readlines()]))

    CLUSTERING = 'AP'
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
            'max_iter': 150,
            'convergence_iter': 15,
            'preference': None,
            'affinity': 'euclidean'
        }
    }
    logger.info('Clustering using %s' % CLUSTERING)

    jsd_scores = compute_jsd_scores(filepath1, filepath2, CLUSTERING, args_dicts[CLUSTERING], target_words)
    with open('{}.csv'.format(outPath), 'w') as f:
        for word, score in jsd_scores.items():
            f.write("{}\t{}\n".format(word, score))


if __name__ == '__main__':
    main()
