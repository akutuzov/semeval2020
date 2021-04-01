# /bin/env python3
# coding: utf-8

import argparse
import logging
from simple_elmo import ElmoModel
from smart_open import open
import json
import time

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Compute LM predictions')
    arg = parser.add_argument
    arg('--elmo', "-e", help='Location of checkpoint files', required=True)
    arg('--input', "-i", help='Input file', required=True)
    arg("--vocab", "-v", help="Path to vocabulary file", required=True)
    arg("--name", "-n", help="Out file prefix (added to the word)", default="_substitutes.json.gz")
    arg("--batch", "-b", help="Batch size", type=int, default=2)

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

    targets = set(lemma_targets)

    model = ElmoModel()

    model.load(args.elmo, full=True, max_batch_size=args.batch)

    target_substitutes = {w: [] for w in targets}

    start = time.time()
    CACHE = 1000
    lines_processed = 0
    lines_cache = []

    with open(data_path, "r") as dataset:
        for line in dataset:
            res = line.strip().split()[:WORD_LIMIT]
            if targets & set(res):
                lines_cache.append(" ".join(res))
            lines_processed += 1
            if len(lines_cache) == CACHE:
                lex_substitutes = model.get_elmo_substitutes(lines_cache)
                for sent in lex_substitutes:
                    for word in sent:
                        if word["word"] in targets:
                            data2add = {el: word[el] for el in word if el != "word"}
                            target_substitutes[word["word"]].append(data2add)
                lines_cache = []
            if lines_processed % 5120 == 0:
                logger.info(f"{data_path}; Lines processed: {lines_processed}")
        if lines_cache:
            logger.debug(f"We fed {len(lines_cache)} sentences")
            lex_substitutes = model.get_elmo_substitutes(lines_cache)
            logger.debug(f"We have {len(lex_substitutes)} sentences")
            for sent in lex_substitutes:
                for word in sent:
                    if word["word"] in targets:
                        data2add = {el: word[el] for el in word if el != "word"}
                        target_substitutes[word["word"]].append(data2add)

    end = time.time()
    processing_time = int(end - start)
    logger.info(f"ELMo substitutes for your input are ready in {processing_time} seconds")
    logger.info("Saving...")

    lemma_substitutes = {}
    for word in target_substitutes:
        lemma = lemma_targets[word]
        if lemma not in lemma_substitutes:
            lemma_substitutes[lemma] = []
        lemma_substitutes[lemma] += target_substitutes[word]

    for word in lemma_substitutes:
        if len(lemma_substitutes[word]) < 1:
            logger.debug(f"No occurrences found for {word}!")
            continue
        outfile = word + args.name
        with open(outfile, "w") as f:
            for occurrence in lemma_substitutes[word]:
                out = json.dumps(occurrence, ensure_ascii=False)
                f.write(out + "\n")
        logger.info(f"Substitutes saved to {outfile}")

