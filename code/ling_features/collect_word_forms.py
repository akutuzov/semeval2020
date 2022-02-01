# python3
# coding: utf-8

import argparse
import logging
from collections import defaultdict
from smart_open import open

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to a CONLL file", required=True)
    arg("--target", "-t", help="Path to target words", required=True)
    arg("--output", "-o", help="Output path (csv)", required=False)

    args = parser.parse_args()

    target_words = {}

    for line in open(args.target, "r"):
        word = line.strip().split("\t")[0]
        pos = None
        if "_" in word:
            word, pos = word.split("_")
        if pos == "nn":
            pos = "NOUN"
        elif pos == "vb":
            pos = "VERB"
        target_words[word] = pos

    forms = defaultdict(list)
    for line in open(args.input, "r"):
        if not line.strip():
            continue
        if line.startswith("# "):
            continue
        (
            identifier,
            form,
            lemma,
            pos,
            xpos,
            feats,
            head,
            rel,
            enh,
            misc,
        ) = line.strip().split("\t")
        if lemma in target_words:
            if target_words[lemma]:
                if pos != target_words[lemma]:
                    continue
            if form not in forms[lemma]:
                forms[lemma].append(form)

    if args.output:
        with open(f"{args.output}.csv", "w") as f:
            for lemma in sorted(forms.keys()):
                f.write(f"{lemma},{','.join(sorted(forms[lemma]))}\n")
    else:
        for lemma in sorted(forms.keys()):
            print(f"{lemma},{','.join(sorted(forms[lemma]))}\n")
