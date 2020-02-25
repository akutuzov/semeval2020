# python3
# coding: utf-8

import sys
import argparse
from smart_open import open
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)
    arg('--outfile', '-o', help='Output file to save embeddings', required=True)
    arg('--vocab', '-v', help='Path to vocabulary file', required=True)
    arg('--batch', '-b', help='ELMo batch size', default=64, type=int)

    args = parser.parse_args()
    data_path = args.input
    batch_size = args.batch
    vocab_path = args.vocab

    vect_dict = {}
    with open(vocab_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            vect_dict[word] = 0

    print('Words to test:', len(vect_dict), file=sys.stderr)
    print('Counting occurrences...:', file=sys.stderr)

    with open(data_path, 'r') as corpus:
        for line in corpus:
            res = line.strip().split()[:500]
            for word in res:
                if word in vect_dict:
                    vect_dict[word] += 1

    # Loading a pre-trained ELMo model:
    # You can call load_elmo_embeddings() with top=True to use only the top ELMo layer
    batcher, sentence_character_ids, elmo_sentence_input, vector_size = load_elmo_embeddings(
        args.elmo, top=True)

    vect_dict = {word: np.zeros((int(vect_dict[word]), vector_size)) for word in vect_dict}
    target_words = set(vect_dict)

    counters = {w: 0 for w in vect_dict}

    # Actually producing ELMo embeddings for our data:
    lines_processed = 0
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        lines_cache = []
        with open(data_path, 'r') as dataset:
            for line in dataset:
                res = line.strip().split()[:500]
                if target_words & set(res):
                    lines_cache.append(res)
                    lines_processed += 1
                if len(lines_cache) == batch_size:
                    elmo_vectors = get_elmo_vectors(sess, lines_cache, batcher,
                                                    sentence_character_ids, elmo_sentence_input)

                    for sent, matrix in zip(lines_cache, elmo_vectors):
                        for word, vector in zip(sent, matrix):
                            if word in vect_dict:
                                vect_dict[word][counters[word], :] = vector
                                counters[word] += 1

                    lines_cache = []
                    if lines_processed % 256 == 0:
                        print('Lines processed:', lines_processed, file=sys.stderr)

    print('Vector extracted. Pruning zeros...', file=sys.stderr)
    vect_dict = {w: vect_dict[w][~(vect_dict[w] == 0).all(1)] for w in vect_dict}

    print('ELMo embeddings for your input are ready', file=sys.stderr)

    np.savez_compressed(args.outfile, **vect_dict)

    print('Vectors saved to', args.outfile, file=sys.stderr)
