from docopt import docopt


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

    # Get the arguments
    args = docopt("""Compare rankings word by word.

    Usage:
        eval.py  <modelAnsPath2> <trueAnsPath2>

    Arguments:
        <modelAnsPath2> = path to tab-separated answer file for Task 2 (lemma + "\t" + corr. coeff.)
        <trueAnsPath2> = path to tab-separated gold answer file for Task 2 (lemma + "\t" + corr. coeff.)
    """)

    modelAnsPath2 = args['<modelAnsPath2>']
    trueAnsPath2 = args['<trueAnsPath2>']

    ranking_diffs = rank_differences(modelAnsPath2, trueAnsPath2)

    for (target, diff) in sorted(ranking_diffs.items(), key=lambda tpl: tpl[1]):
        print('{}\t{}'.format(target, diff))


if __name__ == '__main__':
    main()
