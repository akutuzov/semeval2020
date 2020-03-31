import numpy as np
import matplotlib.pyplot as plt

from docopt import docopt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    """
    Produce PCA plot of a word's contextualised embeddings.
    """

    # Get the arguments
    args = docopt("""Produce PCA plot of a word's contextualised embeddings.

    Usage:
        pca.py <npz1> <npz2> <target> <out>

    Arguments:
        <npz1> = path to npz file containing all embeddings for corpus 1 (for all target words)
        <npz2> = path to npz file containing all embeddings for corpus 2 (for all target words)
        <target> = the target word
        <out> = output path for the PCA plot

    """)

    npz1 = args['<npz1>']
    npz2 = args['<npz2>']
    target = args['<target>']
    out_path = args['<out>']

    usages1 = np.load(npz1)
    usages2 = np.load(npz2)

    try:
        usages1 = usages1[target]
    except KeyError:
        print('No usages of "{}" in corpus 1.'.format(target))
        usages1 = None

    try:
        usages2 = usages2[target]
    except KeyError:
        print('No usages of "{}" in corpus 2.'.format(target))
        usages2 = None

    if usages1 is None and usages2 is None:
        print('No usages of "{}". Stop.'.format(target))
        return

    x = np.concatenate([usages1, usages2], axis=0)

    x = StandardScaler().fit_transform(x)
    x_2d = PCA(n_components=2).fit_transform(x)

    plt.figure(figsize=(15, 15))
    plt.xticks([]), plt.yticks([])
    plt.title("'{}'\n".format(target), fontsize=20)

    usages = [x_2d[:len(usages1), :], x_2d[len(usages1):, :]]
    labels = ['Corpus 1', 'Corpus 2']
    colors = ['b', 'g']

    for matrix, label, color in zip(usages, labels, colors):
        plt.scatter(matrix[:, 0], matrix[:, 1], c=color, s=20)

    plt.legend(labels, prop={'size': 15})

    plt.savefig(out_path)


if __name__ == '__main__':
    main()
