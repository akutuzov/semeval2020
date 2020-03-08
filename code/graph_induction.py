# python3
# coding: utf-8

import os
import string
import logging
import argparse
from time import time
from collections import Counter
from typing import List, Dict, Set
from smart_open import open
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph
from gensim.models import KeyedVectors
from chinese_whispers import chinese_whispers, aggregate_clusters

"""
Produces sense inventories for words in TARGET_WORDS file 
from a static word embedding model in EMBEDDINGS
python3.6 graph_induction.py --emb EMBEDDINGS --eval TARGET_WORDS
Based on https://github.com/uhh-lt/158/blob/master/158_disambiguator/models/graph_induction.py
"""

# Max number of neighbors
verbose = False
LIMIT = 100000
BATCH_SIZE = 2000
GPU_DEVICE = 0


def load_globally(word_vectors_fpath: str, faiss_gpu: bool):
    global wv
    global index_faiss

    print("Loading word vectors from:", word_vectors_fpath)
    tic = time()
    if word_vectors_fpath.endswith(".vec.gz"):
        wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False,
                                               unicode_errors="ignore")
    else:
        wv = KeyedVectors.load(word_vectors_fpath)
    print("Loaded in {} sec.".format(time() - tic))

    wv.init_sims(replace=True)

    if faiss_gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatIP(wv.vector_size)  # build a flat (CPU) index
        index_faiss = faiss.index_cpu_to_gpu(res, GPU_DEVICE,
                                             index_flat)  # make it into a gpu index
        index_faiss.add(wv.syn0norm)  # add vectors to the index
    else:
        index_faiss = faiss.IndexFlatIP(wv.vector_size)
        index_faiss.add(wv.syn0norm)
    return wv


def get_nns(target: str, neighbors_number: int):
    """
    Get neighbors for target word
    :param target: word to find neighbors
    :param neighbors_number: number of neighbors
    :return: list of target neighbors
    """
    target_neighbors = voc_neighbors[target]
    if len(target_neighbors) >= neighbors_number:
        return target_neighbors[:neighbors_number]
    else:
        print("neighbors_number {} is more than precomputed {}".format(neighbors_number,
                                                                       len(target_neighbors)))
        exit(1)


def get_nns_faiss_batch(targets: List, batch_size: int, neighbors_number: int = 50) -> Dict:
    """
    Get neighbors for targets by Faiss with a batch-split.
    :param targets: list of target words
    :param batch_size: how many words to push into Faiss
    :param neighbors_number: number of neighbors
    :return: dict of word -> list of neighbors
    """

    word_neighbors_dict = dict()
    if verbose:
        print("Start Faiss with batches")

    logger.info("Start Faiss with batches")

    for start in range(0, len(targets), batch_size):
        end = start + batch_size

        if verbose:
            print("batch {} to {} of {}".format(start, end, len(targets)))

        logger.info("batch {} to {} of {}".format(start, end, len(targets)))

        batch_dict = get_nns_faiss(targets[start:end], neighbors_number=neighbors_number)
        word_neighbors_dict = {**word_neighbors_dict, **batch_dict}

    return word_neighbors_dict


def get_nns_faiss(targets: List, neighbors_number: int = 200) -> Dict:
    """
    Get nearest neighbors for list of targets without batches.
    :param targets: list of target words
    :param neighbors_number: number of neighbors
    :return: dict of word -> list of neighbors
    """

    numpy_vec = np.array([wv[target] for target in targets])  # Create array of batch vectors
    cur_d, cur_i = index_faiss.search(numpy_vec, neighbors_number + 1)  # Find neighbors

    # Write neighbors into dict
    word_neighbors_dict = dict()
    for word_index, (_D, _I) in enumerate(zip(cur_d, cur_i)):

        # Check if word is punct
        nns_list = []
        for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
            if n > 0:
                nns_list.append((wv.index2word[i], d))

        word_neighbors_dict[targets[word_index]] = nns_list

    return word_neighbors_dict


def in_nns(nns, word: str) -> bool:
    """Check if word is in list of tuples nns."""
    for w, s in nns:
        if word.strip().lower() == w.strip().lower():
            return True

    return False


def get_pair(first, second) -> tuple:
    pair_lst = sorted([first, second])
    sorted_pair = (pair_lst[0], pair_lst[1])
    return sorted_pair


def get_disc_pairs(ego, neighbors_number: int) -> Set:
    pairs = set()

    nns = get_nns(ego, neighbors_number)
    nns_words = [row[0] for row in nns]  # list of neighbors (only words)
    wv_neighbors = np.array([wv[nns_word] for nns_word in nns_words])
    wv_ego = np.array(wv[ego])
    wv_negative_neighbors = (wv_neighbors - wv_ego) * (-1)  # untop vectors

    # find top neighbor for each difference:
    cur_d, cur_i = index_faiss.search(wv_negative_neighbors, 1 + 1)

    # Write down top-untop pairs
    pairs_list_2 = list()
    for word_index, (_D, _I) in enumerate(zip(cur_d, cur_i)):
        for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
            if wv.index2word[i] != ego:  # faiss find either ego-word or untop we need
                pairs_list_2.append((nns_words[word_index], wv.index2word[i]))
                break

    # Filter pairs
    for pair in pairs_list_2:
        if in_nns(nns, pair[1]):
            pairs.add(get_pair(pair[0], pair[1]))

    return pairs


def get_nodes(pairs: Set) -> Counter:
    nodes = Counter()
    for src, dst in pairs:
        nodes.update([src])
        nodes.update([dst])

    return nodes


def list2dict(lst: list) -> Dict:
    return {p[0]: p[1] for p in lst}


def wsi(ego, neighbors_number: int) -> Dict:
    """
    Gets graph of neighbors for word (ego)
    :param ego: word
    :param neighbors_number: number of neighbors
    :return: dict of network and nodes
    """
    tic = time()
    ego_network = Graph(name=ego)

    pairs = get_disc_pairs(ego, neighbors_number)
    nodes = get_nodes(pairs)

    ego_network.add_nodes_from([(node, {'size': size}) for node, size in nodes.items()])

    for r_node in ego_network:
        related_related_nodes = list2dict(get_nns(r_node, neighbors_number))
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if
             rr_node in ego_network],
            reverse=True)[:neighbors_number]

        related_edges = []
        for w, rr_node in related_related_nodes_ego:
            if get_pair(r_node, rr_node) not in pairs:
                related_edges.append((r_node, rr_node, {"weight": w}))
            else:
                # print("Skipping:", r_node, rr_node)
                pass
        ego_network.add_edges_from(related_edges)

    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose:
        print("{}\t{:f} sec.".format(ego, time() - tic))

    return {"network": ego_network, "nodes": nodes}


def draw_ego(graph, show: bool = False, save_fpath: str = ""):
    colors = [1. / graph.node[node]['label'] for node in graph.nodes()]
    sizes = [300. * graph.node[node]['size'] for node in graph.nodes()]

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    nx.draw_networkx(graph, cmap=plt.get_cmap('gist_rainbow'),
                     node_color=colors,
                     font_color='black',
                     node_size=sizes)

    if show:
        plt.show()
    if save_fpath != "":
        plt.savefig(save_fpath)

    fig.clf()


def get_target_words(targetfile) -> List:
    words = set()
    for line in open(targetfile):
        words.add(line.strip())
    words = sorted(words)
    return words


def get_cluster_lines(graph, nodes):
    lines = []
    labels_clusters = sorted(aggregate_clusters(graph).items(),
                             key=lambda e: len(e[1]), reverse=True)
    for label, cluster in labels_clusters:
        scored_words = []
        for word in cluster:
            scored_words.append((nodes[word], word))
        keyword = sorted(scored_words, reverse=True)[0][1]

        lines.append("{}\t{}\t{}\t{}\n".format(graph.name, label, keyword, ", ".join(cluster)))
    if len(lines) == 0:
        lines.append("{}\t{}\t{}\t{}\n".format(graph.name, 1, graph.name, 'None'))
    return lines


def run(embfile=None, eval_vocabulary=None, visualize: bool = True,
        show_plot: bool = False, faiss_gpu: bool = True, outfile=None):
    # Get w2v models paths
    wv_fpath = embfile

    # ensure the word vectors are saved in the fast to load gensim format
    embedding = load_globally(wv_fpath, faiss_gpu)

    # Get list of words for language
    if eval_vocabulary:
        pre_voc = get_target_words(eval_vocabulary)
        voc = [w for w in pre_voc if w in embedding.index2word]
        if len(voc) != len(pre_voc):
            print('%d target words missing from the embeddings:' % len(set(pre_voc) - set(voc)))
            print([w for w in pre_voc if w not in voc])
    else:
        voc = embedding.index2word

    words = {w: None for w in voc}

    print("Vocabulary: {} words".format(len(voc)))

    # Load neighbors for vocabulary (globally)
    global voc_neighbors
    voc_neighbors = get_nns_faiss_batch(embedding.index2word, batch_size=BATCH_SIZE)

    # Init folder for inventory plots
    plt_path = None
    if visualize:
        plt_path = "plots"
        os.makedirs(plt_path, exist_ok=True)

    # perform word sense induction
    for topn in [50]:

        if visualize:
            plt_topn_path = os.path.join(plt_path, str(topn))
            os.makedirs(plt_topn_path, exist_ok=True)
        else:
            plt_topn_path = None

        if verbose:
            print('{} neighbors'.format(topn))

        logger.info("{} neighbors".format(topn))

        with open(outfile, "w") as out:
            out.write("word\tcid\tkeyword\tcluster\n")

        for index, word in enumerate(words):
            if index + 1 == LIMIT:
                print("OUT OF LIMIT {}".format(LIMIT))

                logger.error("OUT OF LIMIT".format(LIMIT))
                break

            if word in string.punctuation:
                print("Skipping word '{}', because it is a punctuation\n".format(word))
                continue

            if verbose:
                print("{} neighbors, word {} of {}, LIMIT = {}\n".format(
                    topn, index + 1, len(words), LIMIT))

            logger.info("{} neighbors, word {} of {}, LIMIT = {}".format(
                topn, index + 1, len(words), LIMIT))

            try:
                words[word] = wsi(word, neighbors_number=topn)
                if visualize:
                    plt_topn_path_word = os.path.join(plt_topn_path, "{}.png".format(word))
                    draw_ego(words[word]["network"], show_plot, plt_topn_path_word)
                lines = get_cluster_lines(words[word]["network"], words[word]["nodes"])
                with open(outfile, "a") as out:
                    for l in lines:
                        out.write(l)

            except KeyboardInterrupt:
                break


def main():
    parser = argparse.ArgumentParser(description='Graph-Vector Word Sense Induction approach.')
    parser.add_argument("--emb", '-e', help="Path to embeddings")
    parser.add_argument("--outfile", '-o', help="Path to output file with senses")
    parser.add_argument("--eval",
                        help="Use only evaluation vocabulary, not all words in the model.",
                        default=None)
    parser.add_argument("--viz", help="Visualize each ego networks.", action="store_true",
                        default=False)
    parser.add_argument("--faiss_gpu", help="Use GPU for faiss", action="store_true", default=False)
    args = parser.parse_args()

    run(embfile=args.emb, eval_vocabulary=args.eval, visualize=args.viz,
        faiss_gpu=args.faiss_gpu, outfile=args.outfile)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
