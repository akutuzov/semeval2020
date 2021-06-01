# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import gzip
from collections import defaultdict

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to a CONLL file in gz format", required=True)
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

    print(target_words)
        
properties = {w: defaultdict(int) for w in target_words}

sentence = []
targets = defaultdict(list)

for line in gzip.open(args.input, "r"):
    if not line.strip():
        continue

    line = line.decode("utf-8") 
    
    if line.startswith('# '):
        for t,ids in targets.items():
            for i in ids:
                for (rel, head) in sentence:
                    if head == i:
                        properties[t][rel] += 1

           
        sentence = []
        targets = defaultdict(list)
            
    else:
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
        ) = line.strip().split('\t')

        
        if lemma in target_words:
            if target_words[lemma]and pos != target_words[lemma]:
                    continue
            else:
                targets[lemma].append(identifier)

       
        sentence.append((rel,head))
        
                
if args.output:
    with open(f"{args.output}_reverse_synt.json", "w") as f:
        out = json.dumps(properties, ensure_ascii=False, indent=4)
        f.write(out)
else:
    print(properties)
