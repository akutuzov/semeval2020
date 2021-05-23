# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np
from scipy.spatial.distance import cosine

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input1", "-i1", help="Path to a JSON file 1", required=True)
    arg("--input2", "-i2", help="Path to a JSON file 2", required=True)
    arg("--output", "-o", help="Output path (tsv)", required=False)
    arg("--threshold", "-t", nargs='?', const=0, help="Minimal percentage to keep a feature", default=0, type=int, required=False)
    
    args = parser.parse_args()

    print("ARGS: %s" %args)
    
    with open(args.input1, "r") as f:
        properties_1 = json.loads(f.read())
            
    with open(args.input2, "r") as f:
        properties_2 = json.loads(f.read())

    assert properties_1.keys() == properties_2.keys()

    words = {w: 0 for w in properties_1.keys()}

    all_features = set()

    for word in words:

        p1 = properties_1[word]
        p2 = properties_2[word]
        features = list(p1.keys() | p2.keys())
        
        prop_count = {k:p1.get(k,0)+p2.get(k,0) for k in features}
        total = sum(prop_count.values())
        features = [f for f in features if prop_count[f]/total*100 > args.threshold]

        all_features.update(features)
                
        #logger.info(word)
        #logger.info(features)

        vector_1 = np.zeros(len(features))
        vector_2 = np.zeros(len(features))
        for nr, feature in enumerate(features):
            try:
                vector_1[nr] = p1[feature]
            except KeyError:
                pass
            try:
                vector_2[nr] = p2[feature]
            except KeyError:
                pass

        
            
        distance = cosine(vector_1, vector_2)
        if np.isnan(distance):
            distance = 0.0  # A word was not present in one of the time periods
        words[word] = distance

        
    #print(all_features)

    if args.output:
        with open(f"{args.output}_graded.tsv", "w") as f:
            for w in words:
                f.write(f"{w}\t{words[w]}\n")

        with open(f"{args.output}_binary.tsv", "w") as f:
            values = sorted(words, key=words.get, reverse=True)
            # threshold = int(len(values) / 2)
            threshold = int(len(values) * 0.43)
            # threshold = 17  # This is for English
            for val in values[:threshold]:
                f.write(f"{val}\t1\n")
            for val in values[threshold:]:
                f.write(f"{val}\t0\n")
