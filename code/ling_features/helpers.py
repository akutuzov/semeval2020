# python3
# coding: utf-8

import numpy as np
import ruptures as rpt
from collections import defaultdict


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
