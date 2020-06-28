import numpy as np
from docopt import docopt
from tqdm import tqdm


def reduce(method, matrix, hdim=768, nlayers=13, skipL0=True):
    """
    :param method: 'top', 'average', 'mid4', 'last4'
    :param matrix: a usage matrix of shape (n_occurrences, n_layers * hdim)
    :param hdim: the dimension of a hidden layer in BERT (default: 768)
    :param nlayers: the number of extracted BERT layers (default: 13)
    :param skipL0: whether to skip the 0th layer, i.e., usually, the embedding layer (default: True)
    :return: the reduced usage matrix of shape (n_occurrences, hdim)
    """
    assert matrix.shape[1] == nlayers * hdim, 'Invalid vector dimensionality: {}'.format(matrix.shape[1])

    # if method == 'top':
    #     return matrix[:, -hdim:]

    if skipL0:
        split = np.split(matrix[:, hdim:], nlayers - 1, axis=1)
    else:
        split = np.split(matrix, nlayers, axis=1)

    if method == 'last':
        return np.array(split[-1])
    elif method == 'average':
        return np.mean(np.array(split), axis=0)
    elif method == 'sum':
        return np.sum(np.array(split), axis=0)
    elif method == 'mean-last4':
        return np.mean(np.array(split[-4:]), axis=0)
    elif method == 'sum-last4':
        return np.sum(np.array(split[-4:]), axis=0)
    elif method == 'mean-mid4':
        mid_layer = int(np.floor((nlayers - bool(skipL0)) / 2))
        return np.mean(np.array(split[mid_layer-1: mid_layer+3]), axis=0)
    elif method == 'sum-mid4':
        mid_layer = int(np.floor((nlayers - bool(skipL0)) / 2))
        return np.sum(np.array(split[mid_layer-1: mid_layer+3]), axis=0)
    else:
        raise ValueError('Invalid method:', method)


def main():
    """
    Aggregate BERT layer vectors into a single-vector contextualised representation.
    """

    # Get the arguments
    args = docopt("""Aggregate BERT layer vectors into a single-vector contextualised representation.

    Usage:
        layer_selection.py <method> <allLayersPath> <outPath>

    Arguments:
        <method> = 'last', 'average', 'mean-mid4', 'sum-mid4', 'mean-last4', 'sum-last4'
        <allLayersPath> = path to .npz file containing a dictionary that maps words to usage matrices
        <outPath> = path to output .npz file with aggregated layers
    """)

    filepath_all = args['<allLayersPath>']
    method = args['<method>']
    out_path = args['<outPath>']

    usage_dict = np.load(filepath_all)
    usage_dict_new = {}

    for w in tqdm(usage_dict):
        # if using Bert-Large, add hdim=1024, nlayers=25 
        usage_dict_new[w] = reduce(method, usage_dict[w])

    np.savez_compressed(out_path, **usage_dict_new)


if __name__ == '__main__':
    main()
