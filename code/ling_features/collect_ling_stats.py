# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to a CONLL file", required=True)
    arg("--target", "-t", help="Path to target words", required=True)
    arg("--output", "-o", help="Output path (json)", required=False)

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

    morph_properties = {w: {} for w in target_words}
    syntax_properties = {w: {} for w in target_words}
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
            if feats not in morph_properties[lemma]:
                morph_properties[lemma][feats] = 0
            morph_properties[lemma][feats] += 1
            if rel not in syntax_properties[lemma]:
                syntax_properties[lemma][rel] = 0
            syntax_properties[lemma][rel] += 1

    if args.output:
        with open(f"{args.output}_morph.json", "w") as f:
            out = json.dumps(morph_properties, ensure_ascii=False, indent=4)
            f.write(out)
        with open(f"{args.output}_synt.json", "w") as f:
            out = json.dumps(syntax_properties, ensure_ascii=False, indent=4)
            f.write(out)
    else:
        print(morph_properties)
        print(syntax_properties)
