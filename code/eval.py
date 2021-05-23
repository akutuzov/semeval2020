import numpy as np
from docopt import docopt
from scipy.stats import spearmanr
import os


def get_ys(model_answers, true_answers):
    """
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: a numpy array for the model scores, and one for the true scores
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
    y_hat, y = [], []
    with open(true_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            y.append(float(score))
            y_hat.append(float(y_hat_tmp[lemma]))

    return np.array(y_hat), np.array(y)


def eval_task1(model_answers, true_answers):
    """
    Computes the Accuracy against the true binary labels as annotated by humans.
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: binary classification accuracy
    """
    y_hat, y = get_ys(model_answers, true_answers)
    accuracy = np.sum(np.equal(y_hat, y)) / len(y)

    return accuracy


def eval_task2(model_answers, true_answers):
    """
    Computes the Spearman's correlation coefficient against the true rank as annotated by humans
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: (Spearman's correlation coefficient, p-value)
    """
    y_hat, y = get_ys(model_answers, true_answers)
    r, p = spearmanr(y_hat, y, nan_policy='omit')

    return r, p


def main():
    """
    Evaluate lexical semantic change detection results.
    """

    # Get the arguments
    args = docopt("""Evaluate lexical semantic change detection results.

    Usage:
        eval.py <modelAnsPath1> <modelAnsPath2> <trueAnsPath1> <trueAnsPath2>

    Arguments:
        <modelAnsPath1> = path to tab-separated answer file for Task 1 (lemma + "\t" + binary score)
        <modelAnsPath2> = path to tab-separated answer file for Task 2 (lemma + "\t" + corr. coeff.)
        <trueAnsPath1> = path to tab-separated gold answer file for Task 1 (lemma + "\t" + binary score)
        <trueAnsPath2> = path to tab-separated gold answer file for Task 2 (lemma + "\t" + corr. coeff.)

    """)

    modelAnsPath1 = args['<modelAnsPath1>']
    modelAnsPath2 = args['<modelAnsPath2>']
    trueAnsPath1 = args['<trueAnsPath1>']
    trueAnsPath2 = args['<trueAnsPath2>']

    if os.path.isfile(modelAnsPath1):
        acc = eval_task1(modelAnsPath1, trueAnsPath1)
        print('Task 1 Accuracy: {:.3f}'.format(acc))
    else:
        print('Task 1 predictions not found: %s' %modelAnsPath1)
        pass

    if os.path.isfile(modelAnsPath2):
        r, p = eval_task2(modelAnsPath2, trueAnsPath2)
        print('Task 2 r: {:.3f}  p: {:.3f}'.format(r, p))
    else:
        print('Task 2 predictions not found: %s' %modelAnsPath2)


if __name__ == '__main__':
    main()
