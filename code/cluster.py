import numpy as np
from docopt import docopt
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def cluster(usage_matrix, algorithm, args_dict):
    """
    :param usage_matrix: a matrix of contextualised word representations of shape (num_usages, model_dim)
    :param algorithm:the clustering algorithm: DBSCAN or Affinity Propagation
    :param args_dict: the sklearn parameters of the chosen clustering algorithm
    :return: n_clusters - the number of clusters in the given usage matrix
             labels - a list of labels, one for each contextualised representation
    """
    if algorithm.lower() not in ['dbscan', 'db', 'affinity', 'affinity propagation', 'ap']:
        raise ValueError('Invalid clustering method:', algorithm)

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

    return n_clusters, labels


def get_num_senses(filepath, algorithm, args_dict):
    """
    :param filepath: path to .nzp file containing a dictionary {lemma: usage matrix for lemma in targets}
    :param algorithm: the clustering algorithm: DBSCAN or Affinity Propagation
    :param args_dict: the sklearn parameters of the chosen clustering algorithm
    :return: num_senses - a dictionary {lemma: num_senses_lemma for lemma in targets}
    """
    usage_dict = np.load(filepath)
    num_senses = {w: 0 for w in usage_dict}

    for w, usage_matrix in tqdm(usage_dict.items()):
        usage_matrix = usage_matrix[:, :768]
        num_senses_w, _ = cluster(usage_matrix, algorithm, args_dict)
        num_senses[w] = num_senses_w

    return num_senses


def main():
    """
    Get number of senses given two diachronic sets of usage representations.
    """

    # Get the arguments
    args = docopt("""Get number of senses for two sets of usage representations.

    Usage:
        cluster.py <representationsFile1> <representationsFile2> <outPath>

    Arguments:
        <representationsFile1> = path to .npz file containing a dictionary that maps words to usage matrices (corpus1)
        <representationsFile2> = path to .npz file containing a dictionary that maps words to usage matrices (corpus2)
        <outPath> = output filepath *without extension* for csv files with number of senses 
                    (format: 'lemma num_senses_in_corpus1 num_senses_in_corpus2')
    """)

    filepath1 = args['<representationsFile1>']
    filepath2 = args['<representationsFile2>']
    outPath = args['<outPath>']

    args_dicts = {
        'DB': {
            'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean', 'algorithm': 'auto', 'leaf_size': 30, 'p': None,
        },
        'AP': {
            'damping': 0.5, 'max_iter': 200, 'convergence_iter': 15, 'preference': None, 'affinity': 'euclidean'
        }
    }

    for clustering in ['DB', 'AP']:
        print('>>', clustering, 'clustering')
        num_senses_1 = get_num_senses(filepath1, clustering, args_dicts[clustering])
        num_senses_2 = get_num_senses(filepath2, clustering, args_dicts[clustering])
        assert len(num_senses_1) == len(num_senses_2)

        with open('{}.{}.csv'.format(outPath, clustering), 'w') as f:
            for w in num_senses_1:
                f.write('\t{} {:d} {:d}\n'.format(w, num_senses_1[w], num_senses_2[w]))


if __name__ == '__main__':
    main()
