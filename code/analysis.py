#! python3
# coding: utf-8

from smart_open import open
import numpy as np
import os
import sys


directory = sys.argv[1]

langs = [f for f in os.listdir(directory)]

for lang in langs:
    classes = []
    infile = os.path.join(directory, lang)
    for line in open(infile, 'r'):
        word, label = line.strip().split('\t')
        classes.append(label.strip())
    shifts = classes.count('1')
    ratio = shifts / len(classes)
    print(lang, round(ratio, 2))



