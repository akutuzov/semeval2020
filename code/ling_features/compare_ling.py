# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from collections import defaultdict
import ruptures as rpt

groups = json.load(open("../../data/features/synt_groups.json", "r"))
feature_to_group = {}
for k, v in groups.items():
    for f in v:
        feature_to_group[f] = k



def detect_change_point(sequence, n_chp=1):
    """
    Detects the indices of change points in a sequence of values
    """
    sequence = np.array(sequence)
    algo = rpt.Dynp(model="rbf", jump=1).fit(sequence)
    chp_index, length = algo.predict(n_bkps=n_chp)
    return chp_index


def synt_group(properties, filtering):
    informative = [
        "nominal_function",
        "function",
        "modifier",
        "nominal_modifier",
        "core_nominals",
        "nominal_dependents",
    ]

    new_properties = defaultdict(int)
    for current_feature, count in properties.items():
        group = feature_to_group[current_feature.split(":")[0]]
        if filtering == "group":
            new_properties[group] += count
        elif filtering == "partial":
            group = feature_to_group[current_feature.split(":")[0]]
            if group in informative:
                new_properties[current_feature] = count
            else:
                new_properties[group] += count
        elif filtering == "delete":
            if group in informative:
                new_properties[current_feature] = count
        else:
            raise NotImplementedError
    return new_properties

def feature_separation(word_properties):
    properties = defaultdict(int)
    for el in word_properties:
        for feat in el.split("|"):
            properties[feat] += word_properties[el]
                                                
    return properties


def print_results(words, output, changepoint):
    with open(f"{output}_graded.tsv", "w") as f:
            for w in words:
                f.write(f"{w}\t{words[w]}\n")

    with open(f"{output}_binary.tsv", "w") as f:
            values = sorted(words, key=words.get, reverse=True)
            if changepoint == "automatic":
                threshold = detect_change_point([words[el] for el in values]) + 1
                #logger.info(f"Change point found at {threshold}")
            elif changepoint == "half":
                threshold = int(len(values) / 2)
            elif changepoint == "semeval":
                threshold = int(len(values) * 0.43)
            for val in values[:threshold]:
                f.write(f"{val}\t1\n")
            for val in values[threshold:]:
                f.write(f"{val}\t0\n")

    
def collect_word_properties(properties):
    props = defaultdict(lambda: defaultdict(int))
    for features, count in properties.items():
        separate_features = features.split("|")
        for feat in separate_features:
            try:
                k, v = feat.split("=")
            except:
                continue
            else:
                props[k][v] += count
    return props

def find_features(p1, p2, threshold):
    features = list(p1.keys() | p2.keys())
    prop_count = {k: p1.get(k, 0) + p2.get(k, 0) for k in features}
    total = sum(prop_count.values())
    return [f for f in features if prop_count[f] / total * 100 > threshold]


def compute_distance(vector_1, vector_2, distance_type):
    if distance_type == "cos":
        return cosine(vector_1, vector_2)
    elif distance_type == "jsd":
        return jensenshannon(vector_1, vector_2)
    else:
        raise NotImplementedError("Unknown distance: %s" % args.distance)

    
def make_vectors(features, p1, p2):
        
    vector_1 = np.zeros(len(features))
    vector_2 = np.zeros(len(features))
    
    for nr, feature in enumerate(features):
        vector_1[nr] = p1.get(feature,0)
        vector_2[nr] = p2.get(feature,0)

    return vector_1, vector_2


def compute_distance_from_common_features(p1, p2, threshold, distance_type):
    features = find_features(p1, p2, threshold)
    vector_1, vector_2 = make_vectors(features, p1, p2)
    return compute_distance(vector_1, vector_2, distance_type)

    
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
    arg(
        "--threshold",
        "-t",
        nargs="?",
        const=0,
        help="Minimal percentage to keep a feature",
        default=0,
        type=int,
        required=False,
    )
    arg(
        "--filtering",
        "-f",
        help="Organizing syntactic features according to UD classification: "
        "'group' - grouping all, 'delete' - deleting non-informative, "
        "'partial' - grouping non-informative",
        choices=["group", "partial", "delete", "none"],
        default="none",
    )
    arg(
        "--distance",
        "-d",
        help="Choose between cosine and jsd",
        choices=["cos", "jsd"],
        default="cos",
    )
    arg(
        "--separation",
        "-s",
        help="Morphological feature separation by |",
        choices=["yes", "no", "2step"],
        default="no",
    )
    arg(
        "--agregate",
        "-a",
        help="If separation is '2step' distances are computed separately and then aggregated using either maximum or average",
        choices=["max", "avg"],
        default="max",
    )
    arg(
        "--added_features1",
        "-af1",
        help="Path to JSON file to add syntax features when separation is 2step",
    )
    arg(
        "--added_features2",
        "-af2",
        help="Path to JSON file to add syntax features when separation is 2step",
    )
    
    arg(
        "--changepoint",
        "-cp",
        help="How to detect the change point in distance sequences?",
        choices=["semeval", "half", "automatic"],
        default="automatic",
    )

    
    args = parser.parse_args()

    properties_1 = json.load(open(args.input1, "r"))
    properties_2 = json.load(open(args.input2, "r"))

    if args.added_features1:
        added_features1 = json.load(open(args.added_features1))
        added_features2 = json.load(open(args.added_features2))
        
    assert properties_1.keys() == properties_2.keys()
    words = {w: 0 for w in properties_1.keys()}

    for word in words:
        if args.separation == "2step":

                distance = {}
                
                p1 = collect_word_properties(properties_1[word])
                p2 = collect_word_properties(properties_2[word])
                
                feature_classes = list(p1.keys() | p2.keys())
                
                for f_class in feature_classes:
                    distance[f_class] =\
                                    compute_distance_from_common_features(p1[f_class],
                                                                          p2[f_class],
                                                                          args.threshold,
                                                                          args.distance)
                    
                if args.added_features1:
                    a1 = added_features1[word]
                    a2 = added_features2[word]
                    distance["added"] =\
                                   compute_distance_from_common_features(a1,
                                                                         a2,
                                                                         args.threshold,
                                                                         args.distance)
                        
                distance = [d for d in distance.values() if not np.isnan(d)]
        
                if distance:
                    if args.agregate == "max":
                        words[word] = max(distance)
                    elif args.agregate == "avg":
                        words[word] = np.mean(distance)
                else:
                    # empty, e.g. german
                    words[word] = np.nan

        else:
                if args.separation == "yes":
                    p1 = feature_separation(properties_1[word])
                    p2 = feature_separation(properties_2[word])           
                    
                else:
                    p1 = properties_1[word]
                    p2 = properties_2[word]
        
                if args.filtering != "none":
                    p1 = synt_group(p1, args.filtering)
                    p2 = synt_group(p2, args.filtering)

                distance = compute_distance_from_common_features(p1,
                                                                 p2,
                                                                 args.threshold,
                                                                 args.distance)     
                if np.isnan(distance):
                    distance = 0.0  # A word was not present in one of the time periods
                words[word] = distance

    if args.output:
        print_results(words, args.output, args.changepoint)
    else:
        logger.info(words)
