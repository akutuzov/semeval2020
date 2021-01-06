# python3
# coding: utf-8

import argparse
import numpy as np
from smart_open import open
from simple_elmo import ElmoModel
import logging
import time

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to input text", required=True)
    arg("--elmo", "-e", help="Path to ELMo model", required=True)
    arg("--outfile", "-o", help="Output file to save embeddings", required=True)
    arg("--vocab", "-v", help="Path to vocabulary file", required=True)
    arg("--batch", "-b", help="ELMo batch size", default=64, type=int)
    arg(
        "--layers",
        "-l",
        help="What layers to use",
        default="top",
        choices=["top", "average", "all"],
    )
    arg(
        "--warmup",
        "-w",
        help="Warmup before extracting?",
        default="yes",
        choices=["yes", "no"],
    )

    args = parser.parse_args()
    data_path = args.input
    batch_size = args.batch
    vocab_path = args.vocab
    WORD_LIMIT = 400

    vect_dict = {}
    with open(vocab_path, "r") as f:
        for line in f.readlines():
            word = line.strip()
            vect_dict[word] = 0

    logger.info(f"Words to test: {len(vect_dict)}")
    logger.info("Counting occurrences...")

    wordcount = 0
    with open(data_path, "r") as corpus:
        for line in corpus:
            res = line.strip().split()[:WORD_LIMIT]
            for word in res:
                if word in vect_dict:
                    vect_dict[word] += 1
                    wordcount += 1
    logger.info(f"Total occurrences of target words: {wordcount}")
    logger.info(vect_dict)

    # Loading a pre-trained ELMo model:
    model = ElmoModel()
    model.load(args.elmo, max_batch_size=batch_size)

    vect_dict = {
        word: np.zeros((int(vect_dict[word]), model.vector_size)) for word in vect_dict
    }
    target_words = set(vect_dict)

    counters = {w: 0 for w in vect_dict}

    # Actually producing ELMo embeddings for our data:
    start = time.time()

    CACHE = 12800

    lines_processed = 0
    lines_cache = []
    with open(data_path, "r") as dataset:
        for line in dataset:
            res = line.strip().split()[:WORD_LIMIT]
            if target_words & set(res):
                lines_cache.append(res)
                lines_processed += 1
            if len(lines_cache) == CACHE:
                elmo_vectors = model.get_elmo_vectors(lines_cache, layers=args.layers)
                for sent, matrix in zip(lines_cache, elmo_vectors):
                    for word, vector in zip(sent, matrix):
                        if word in vect_dict:
                            vect_dict[word][counters[word], :] = vector
                            counters[word] += 1
                lines_cache = []
                if lines_processed % 256 == 0:
                    logger.info(f"{data_path}; Lines processed: {lines_processed}")
        if lines_cache:
            elmo_vectors = model.get_elmo_vectors(lines_cache, layers=args.layers)
            for sent, matrix in zip(lines_cache, elmo_vectors):
                for word, vector in zip(sent, matrix):
                    if word in vect_dict:
                        vect_dict[word][counters[word], :] = vector
                        counters[word] += 1
    end = time.time()
    processing_time = int(end - start)
    print(f"ELMo embeddings for your input are ready in {processing_time} seconds")

    logger.info("Vector extracted. Pruning zeros...")
    vect_dict = {w: vect_dict[w][~(vect_dict[w] == 0).all(1)] for w in vect_dict}

    logger.info("Saving...")

    np.savez_compressed(args.outfile, **vect_dict)

    logger.info(f"Vectors saved to {args.outfile}")
