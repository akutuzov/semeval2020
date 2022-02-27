import gensim
import numpy as np


def smart_procrustes_align_gensim(base_embed: gensim.models.KeyedVectors,
                                  other_embed: gensim.models.KeyedVectors):
    """
    This code, taken from
    https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and modified,
    uses procrustes analysis to make two word embeddings compatible.
    :param base_embed: first embedding
    :param other_embed: second embedding to be changed
    :return other_embed: changed embedding
    """
    base_embed.init_sims()
    other_embed.init_sims()

    shared_vocab = list(set(base_embed.wv.vocab.keys()).intersection(other_embed.wv.vocab.keys()))

    base_idx2word = {num: word for num, word in enumerate(base_embed.wv.index2word)}
    other_idx2word = {num: word for num, word in enumerate(other_embed.wv.index2word)}

    base_word2idx = {word: num for num, word in base_idx2word.items()}
    other_word2idx = {word: num for num, word in other_idx2word.items()}

    base_shared_indices = [base_word2idx[word] for word in shared_vocab]
    other_shared_indices = [other_word2idx[word] for word in shared_vocab]

    base_vecs = base_embed.wv.syn0norm
    other_vecs = other_embed.wv.syn0norm

    base_shared_vecs = base_vecs[base_shared_indices]
    other_shared_vecs = other_vecs[other_shared_indices]

    m = other_shared_vecs.T @ base_shared_vecs
    u, _, v = np.linalg.svd(m)
    ortho = u @ v

    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.syn0norm = other_embed.wv.syn0 = other_embed.wv.syn0norm.dot(ortho)

    return other_embed
