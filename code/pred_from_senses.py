# python3
# coding: utf-8

import argparse
from smart_open import open

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st file with the senses', required=True)
    arg('--input1', '-i1', help='Path to 2nd file with the senses', required=True)
    arg('--threshold', '-t', help='Optional degree of sense change to predict a shift', type=int,
        default=1)
    arg('--strength', help='Predict strength of shift?', default=False, type=bool)

    args = parser.parse_args()

    words0 = {}
    words1 = {}

    for line in open(args.input0, 'r').readlines()[1:]:
        res = line.strip().split('\t')
        word = res[0].strip()
        if word not in words0:
            words0[word] = 0
        words0[word] += 1

    for line in open(args.input1, 'r').readlines()[1:]:
        res = line.strip().split('\t')
        word = res[0].strip()
        if word not in words1:
            words1[word] = 0
        words1[word] += 1

    common = set(words0) & set(words1)
    if set(words0) != set(words1):
        all_words = set(words0).union(set(words1))
        diff = all_words - common
        for word in diff:
            nword = word.split('_')[0]
            if args.strength == True:
                print('\t'.join([nword, '10']))
            else:
                print('\t'.join([nword, '1']))

    for word in common:
        nword = word.split('_')[0]
        if args.strength == True:
            strength = abs(words0[word] - words1[word])
            print('\t'.join([nword, str(strength)]))
        else:
            if words0[word] != words1[word]:
                if args.threshold != 0:
                    delta = abs(words0[word] - words1[word])
                    if delta > args.threshold:
                        print('\t'.join([nword, '1']))
                    else:
                        print('\t'.join([nword, '0']))
                    continue
                print('\t'.join([nword, '1']))
            else:
                print('\t'.join([nword, '0']))
