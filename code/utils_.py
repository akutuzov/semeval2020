import pickle
from scipy.sparse import csr_matrix, load_npz, save_npz
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Space(object):
    """
    Load and save Space objects.
    """

    def __init__(self, path=None, matrix=csr_matrix([]), rows=[], columns=[]):
        """
        Can be either initialized (i) by providing a path, (ii) by providing a matrix, rows and
        columns, or (iii) by providing neither, then an empty instance is created
        `path` should be path to a matrix in npz format, expects rows and columns in same folder
        at '[path]_rows' and '[path]_columns'
        `rows` list with row names
        `columns` list with column names
        """

        if path is not None:
            # Load matrix
            matrix = load_npz(path)
            # Load rows
            with open(path + '_rows', 'rb') as f:
                rows = pickle.load(f)
            # Load columns
            with open(path + '_columns', 'rb') as f:
                columns = pickle.load(f)

        row2id = {r: i for i, r in enumerate(rows)}
        id2row = {i: r for i, r in enumerate(rows)}
        column2id = {c: i for i, c in enumerate(columns)}
        id2column = {i: c for i, c in enumerate(columns)}

        self.matrix = csr_matrix(matrix)
        self.rows = rows
        self.columns = columns
        self.row2id = row2id
        self.id2row = id2row
        self.column2id = column2id
        self.id2column = id2column

    def save(self, path):
        """
        `path` saves matrix at path in npz format, saves rows and columns as pickled lists in
        same folder at '[path]_rows' and '[path]_columns'
        """

        # Save matrix
        with open(path, 'wb') as f:
            save_npz(f, self.matrix)
            # Save rows
        with open(path + '_rows', 'wb') as f:
            pickle.dump(self.rows, f)
        # Save columns
        with open(path + '_columns', 'wb') as f:
            pickle.dump(self.columns, f)


def calc_coeffs(embfile, method="centroid"):
    array = np.load(embfile)
    logger.info('Loaded an array of %d entries from %s' % (len(array), embfile))
    words = {}
    for word in array:
        if array[word].shape[0] < 3:
            logger.info('%s omitted because of low frequency: %d' % (word, array[word].shape[0]))
            continue
        if method == 'pairwise':
            var_coeff = pairwise_diversity(array[word])
        else:
            var_coeff = diversity(array[word])
        words[word] = var_coeff
    logger.info('Variation coefficients produced')
    return words


def diversity(matrix):
    mean = np.average(matrix, axis=0)
    distances = [cosine(v, mean) for v in matrix]
    return np.mean(distances)


def pairwise_diversity(matrix):
    distances = pdist(matrix, metric='cosine')
    return np.mean(distances)


def diversity_ax(matrix):
    stds = np.std(matrix, axis=0)
    return np.mean(stds)
