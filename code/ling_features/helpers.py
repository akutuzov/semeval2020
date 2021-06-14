# python3
# coding: utf-8

import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from collections import defaultdict
import ruptures as rpt
import matplotlib.pyplot as plt


def detect_change_point(sequence, n_chp=1):
    """
    Detects the indices of change points in a sequence of values
    """
    sequence = np.array(sequence)
    algo = rpt.Dynp(model="rbf", jump=1).fit(sequence)
    chp_index, length = algo.predict(n_bkps=n_chp)
    return chp_index


def synt_group(properties, filtering, feature_to_group):
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
            # logger.info(f"Change point found at {threshold}")
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
            except ValueError:
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
        dist = cosine(vector_1, vector_2)
        if np.isnan(dist):
            return 0.0
        else:
            return dist
    elif distance_type == "jsd":
        return jensenshannon(vector_1, vector_2)
    else:
        raise NotImplementedError(f"Unknown distance: {distance_type}")


def make_vectors(features, p1, p2):
    vector_1 = np.zeros(len(features))
    vector_2 = np.zeros(len(features))

    for nr, feature in enumerate(features):
        vector_1[nr] = p1.get(feature, 0)
        vector_2[nr] = p2.get(feature, 0)

    return vector_1, vector_2


def compute_distance_from_common_features(p1, p2, threshold, distance_type):
    features = find_features(p1, p2, threshold)
    vector_1, vector_2 = make_vectors(features, p1, p2)
    return compute_distance(vector_1, vector_2, distance_type)


def cat_plot(values, labels):
    pos = np.arange(len(labels))
    plt.bar(pos, values, tick_label=labels, color=['black', 'red', 'green', 'blue', 'cyan', "pink", "tomato", "gray", "brown", "darkviolet"])
    plt.legend(loc="best")
    plt.show()
