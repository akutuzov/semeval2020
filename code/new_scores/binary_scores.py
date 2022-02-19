# python3
# coding: utf-8


import argparse
import numpy as np
import logging
import ruptures as rpt


if __name__ == '__main__':
    """
    Convert continuous change scores into binary scores.
    """

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to csv/tsv file with continuous change scores', required=True)
    arg('--output', '-o', help='Output path (tsv)', required=False)
    arg('--changepoint', '-c', default='automatic', choices=['automatic', 'half', 'semeval'],
        help='The strategy for change point detection: automatic, half list, '
             'or semeval threshold (43% of the list).')

    args = parser.parse_args()


    def detect_change_point(sequence, n_chp=1):
        """
        Detects the indices of change points in a sequence of values
        """
        sequence = np.array(sequence)
        algo = rpt.Dynp(model="rbf", jump=1).fit(sequence)
        chp_index, length = algo.predict(n_bkps=n_chp)
        return chp_index

    def load_scores(path):
        scores = {}
        with open(path, 'r', encoding='utf-8') as f_in:
            for line in f_in.readlines():
                line = line.strip().split('\t')
                target, score = line[0], line[1]
                scores[target] = float(score)
        return scores


    continuous_scores = load_scores(args.input)
    logger.info("Loaded an array of {len(continuous_scores)} entries from {args.input}")

    logger.debug(continuous_scores)

    with open(f"{args.output}", "w") as f:
        values = sorted(continuous_scores, key=continuous_scores.get, reverse=True)
        logger.debug(values)
        if args.changepoint == "automatic":
            threshold = detect_change_point([continuous_scores[el] for el in values]) + 1
        elif args.changepoint == "half":
            threshold = int(len(values) / 2)
        elif args.changepoint == "semeval":
            threshold = int(len(values) * 0.43)
        for val in values[:threshold]:
            f.write(f"{val}\t1\n")
        for val in values[threshold:]:
            f.write(f"{val}\t0\n")

        logger.info(f"Change point after {threshold} out of {len(values)} target words")

        logger.info(f'Binary scores saved to: {args.output}')
