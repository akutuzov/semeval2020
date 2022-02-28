import argparse


def get_ys(model_answers, true_answers):
    """
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: a dictionary for the model scores, and one for the true scores
    """
    y_hat_tmp = {}
    errors = 0
    with open(model_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            if score == 'nan':
                errors += 1
            y_hat_tmp[lemma] = score

    if errors:
        print('Found %d NaN predictions' % errors)

    y_hat, y = {}, {}
    with open(true_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            y[lemma] = float(score)
            y_hat[lemma] = float(y_hat_tmp[lemma])

    return y_hat, y


def rank_differences(model_answers, true_answers):
    """
    Computes the Spearman's correlation coefficient against the true rank as annotated by humans
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: dictionary from target to rank difference
    """
    Y_hat, Y = get_ys(model_answers, true_answers)
    assert len(Y_hat) == len(Y)

    Y_hat_ranks = {}
    for rank, (target, y_hat) in enumerate(sorted(Y_hat.items(), key=lambda kv: kv[1])):
        Y_hat_ranks[target] = rank

    Y_ranks = {}
    for rank, (target, y) in enumerate(sorted(Y.items(), key=lambda kv: kv[1])):
        Y_ranks[target] = rank

    diffs = {}
    for target in Y_ranks:
        diffs[target] = Y_hat_ranks[target] - Y_ranks[target]

    return diffs


def main():
    """
    Compare rankings word by word.
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--predictions", "-p", required=True,
        help='Path to tab-separated answer file for Task 2 (lemma + "\t" + corr. coeff.)')
    arg("--gold_ranking", "-g", required=True,
        help='Path to tab-separated gold answer file for Task 2 (lemma + "\t" + corr. coeff.)')
    arg("--threshold", "-t", default=0.2, type=float, required=False,
        help="Percentage of the ranking differences to consider incorrect. E.g., with default value 0.2, "
             "the lowest 20% of the negative differences will be labelled as false positives; and the "
             "highest 20% of the positive differences will be labelled as false negatives.")

    args = parser.parse_args()

    pred, gold = get_ys(args.predictions, args.gold_ranking)

    ranking_diffs = rank_differences(args.predictions, args.gold_ranking)

    n_20pc = int(len(set(ranking_diffs.values())) * args.threshold)
    unique_diffs = set()
    i = 0
    i_inv = len(set(ranking_diffs.values()))

    sep_string = '-' * 50
    print(f'False positives\n{sep_string}')

    for (target, diff) in sorted(ranking_diffs.items(), key=lambda tpl: tpl[1]):
        if diff not in unique_diffs:
            unique_diffs.add(diff)
            i += 1
            i_inv -= 1

        if i == n_20pc:
            print(f'{sep_string}\n')

        if i_inv == n_20pc:
            print(f'\nFalse negatives\n{sep_string}')

        print('{:25}\t{:.2f}\t{:.2f}\t{}'.format(target, gold[target], pred[target], diff))

    print(sep_string)

if __name__ == '__main__':
    main()
