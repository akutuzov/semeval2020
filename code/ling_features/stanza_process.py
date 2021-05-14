# /bin/env python3
# coding: utf-8

import sys
import stanza
from stanza.utils.conll import CoNLL

lang = sys.argv[1]  # State the segmenter model (en, de, ru, etc)

nlp = stanza.Pipeline(lang, processors="tokenize,mwt,pos,lemma,depparse")

stack = []

for line in sys.stdin:
    doc = nlp(line.strip())
    dicts = doc.to_dict()
    conll = CoNLL.convert_dict(dicts)
    for sentence in conll:
        for token in sentence:
            print("\t".join(token))
        print()
