# python3
# coding: utf-8

import argparse
from smart_open import open
from elmo_helpers import *
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)
    arg('--outfile', '-o', help='Output file to save embeddings', required=True)
    arg('--vocab', '-v', help='Path to vocabulary file', required=True)
    arg('--batch', '-b', help='ELMo batch size', default=64, type=int)
    arg('--top', '-t', help='Use only top layer?', default=True, type=bool)

    args = parser.parse_args()
    data_path = args.input
    batch_size = args.batch
    vocab_path = args.vocab
    WORD_LIMIT = 500

    vect_dict = {}
    with open(vocab_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            vect_dict[word] = 0

    logger.info('Words to test: %d' % len(vect_dict))
    logger.info('Counting occurrences...')

    wordcount = 0
    with open(data_path, 'r') as corpus:
        for line in corpus:
            res = line.strip().split()[:WORD_LIMIT]
            for word in res:
                if word in vect_dict:
                    vect_dict[word] += 1
                    wordcount += 1
    logger.info('Total occurrences of target words: %d' % wordcount)
    logger.info(vect_dict)

    # Loading a pre-trained ELMo model:
    # You can call load_elmo_embeddings() with top=True to use only the top ELMo layer
    batcher, sentence_character_ids, elmo_sentence_input, vector_size = load_elmo_embeddings(
        args.elmo, top=args.top)

    vect_dict = {word: np.zeros((int(vect_dict[word]), vector_size)) for word in vect_dict}
    target_words = set(vect_dict)

    counters = {w: 0 for w in vect_dict}

    # Actually producing ELMo embeddings for our data:
    lines_processed = 0
    with tf.compat.v1.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.compat.v1.global_variables_initializer())

        lines_cache = []
        with open(data_path, 'r') as dataset:
            for line in dataset:
                res = line.strip().split()[:WORD_LIMIT]
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
                        logger.info('%s; Lines processed: %d', data_path, lines_processed)

    logger.info('Vector extracted. Pruning zeros...')
    vect_dict = {w: vect_dict[w][~(vect_dict[w] == 0).all(1)] for w in vect_dict}

    logger.info('ELMo embeddings for your input are ready. Saving...')

    np.savez_compressed(args.outfile, **vect_dict)

    logger.info('Vectors saved to %s' % args.outfile)
