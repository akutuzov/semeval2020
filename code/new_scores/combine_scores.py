# python3
# coding: utf-8


import numpy as np
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st csv file with semantic change scores', required=True)
    arg('--input1', '-i1', help='Path to 2nd csv file with semantic change scores', required=True)
    arg('--output', '-o', help='Output path (csv)', required=False)
    arg('--mode', '-m', help='The combination strategy', default='geometric',
        choices=['sum', 'product', 'arithmetic', 'geometric'])

    args = parser.parse_args()

    def load_scores(path):
        scores = {}
        with open(path, 'r', encoding='utf-8') as f_in:
            for line in f_in.readlines():
                line = line.strip().split('\t')
                target_word, score = line[0], line[1]
                scores[target_word] = float(score)
        return scores

    scores0 = load_scores(args.input0)
    scores1 = load_scores(args.input1)

    targets = sorted(scores0.keys())
    if targets != sorted(scores1.keys()):
        logger.error('The two target lists are different.')

    combined_scores = {}
    for target in targets:
        if args.mode == 'sum':
            combined_scores[target] = scores0[target] + scores1[target]
        elif args.mode == 'product':
            combined_scores[target] = scores0[target] * scores1[target]
        elif args.mode == 'arithmetic':
            combined_scores[target] = (scores0[target] + scores1[target]) / 2
        elif args.mode == 'geometric':
            combined_scores[target] = np.sqrt(scores0[target] * scores1[target])

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for target in targets:
                print(f'{target}\t{str(combined_scores[target])}', file=f_out)
        logger.info(f'Combined scores saved to: {args.output}')
    else:
        for target in targets:
            logger.info(f'{target}\t{str(combined_scores[target])}')
