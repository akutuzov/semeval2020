# python3
# coding: utf-8

import argparse
import numpy as np
import logging
from gensim import utils
from smart_open import open
from sklearn.decomposition import PCA


def save_word2vec_format(fname, vocab, vectors, binary=False):
    """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        vocab : dict
            The vocabulary of words with their ranks
        vectors : numpy.array
            The vectors to be stored
        binary : bool
            If True, the data wil be saved in binary word2vec format,
            else it will be saved in plain text.
        """
    if not (vocab or vectors):
        raise RuntimeError('no input')
    total_vec = len(vocab)
    vector_size = vectors.shape[1]
    logger.info('storing %dx%d projection weights into %s' % (total_vec, vector_size, fname))
    assert (len(vocab), vector_size) == vectors.shape
    with open(fname, 'wb') as fout:
        fout.write(utils.to_utf8('%s %s\n' % (total_vec, vector_size)))
        position = 0
        for element in sorted(vocab, key=lambda entry: vocab[entry]):
            row = vectors[position]
            if binary:
                row = row.astype(np.float32)
                fout.write(utils.to_utf8(element) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8('%s %s\n' % (element, ' '.join(repr(val) for val in row))))
            position += 1
    return fname


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to npz file with the embeddings', required=True)
    arg('--outfile', '-o', help='Save embeddings in the word2vec format?')
    parser.add_argument('--mode', '-m', default='mean', choices=['mean', 'pca'])

    args = parser.parse_args()
    data_path0 = args.input

    array = np.load(data_path0)
    logger.info('Loaded an array of %d entries from %s' % (len(array), data_path0))

    dimensionality = array[array.files[0]].shape[1]
    words = sorted(array.keys())
    processed_vectors = np.zeros((len(words), dimensionality))

    vocabulary = {}
    for nr, word in enumerate(words):
        w_vectors = array[word]
        vocabulary[word] = nr
        if w_vectors.shape[0] < 3:
            logger.info('%s has low frequency: %d; emitting a zero vector'
                        % (word, w_vectors.shape[0]))
            processed_vectors[nr, :] = np.zeros(w_vectors.shape[1])
            continue
        if args.mode == 'pca':
            w_vectors = (w_vectors - np.mean(w_vectors, 0)) / np.std(w_vectors, 0)
            pca = PCA(n_components=3)
            analysis = pca.fit(w_vectors)
            processed_vectors[nr, :] = analysis.components_[0]
        else:
            processed_vectors[nr, :] = np.average(w_vectors, axis=0)

    if args.outfile:
        save_word2vec_format(args.outfile, vocabulary, processed_vectors)
