# /bin/env python3
# coding: utf-8

import argparse
import logging
from smart_open import open

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Compute LM predictions')
    arg = parser.add_argument
    arg('--input', "-i", help='Input file', required=True)
    arg("--vocab", "-v", help="Path to vocabulary file", required=True)

    args = parser.parse_args()

    data_path = args.input
    vocab_path = args.vocab

    lemma_targets = {}
    with open(vocab_path, "r") as f:
        for line in f.readlines():
            wordforms = line.strip().split(",")
            lemma = wordforms[0]
            for word in wordforms:
                lemma_targets[word] = lemma

    WORD_LIMIT = 400

    logger.info(f"Word forms to test: {len(lemma_targets)}")
    logger.info("Counting occurrences...")

    wordcount = 0
    with open(data_path, "r") as corpus:
        for line in corpus:
            res = line.strip().split()[:WORD_LIMIT]
            for word in res:
                if word in lemma_targets:
                    # targets[word] += 1
                    wordcount += 1
    logger.info(f"Total occurrences of target words: {wordcount}")
    logger.info(lemma_targets)