import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys


def get_prediction(path):
    pred = pd.read_csv(path, sep="\t", header=None, index_col=0)
    if not len(pred.columns) == 1:
        return
    pred.columns = [os.path.basename(path).replace(".tsv", "")]
    return pred

def get_correlations(predictions, method, out):
    predictions = [get_prediction(os.path.join(input_dir, p)) for p in predictions]
    predictions = pd.concat(predictions, axis=1)
    corr = predictions.corr(method=method)
    corr.to_csv(out, sep="\t")
    print(out)
    return corr

def plot_correlations(corr, out):
    plt.figure()
    plot = sns.heatmap(corr, 
                       xticklabels=corr.columns.values,
                       yticklabels=corr.columns.values,
                       cmap="BuPu")
    plot.figure.tight_layout()
    plt.savefig(out)

    print(out)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    predictions = sorted([p for p in os.listdir(input_dir) if p.endswith('tsv')])

    graded = [p for p in predictions if 'binary' not in p]
    binary = [p for p in predictions if 'binary' in p]

    graded_corr = get_correlations(graded,
                                  method = 'spearman',
                                  out = os.path.join(input_dir, "graded_cor.tsv"))
    binary_corr = get_correlations(binary,
                                  method = 'pearson',
                                  out = os.path.join(input_dir, "binary_cor.tsv"))



    plot_correlations(graded_corr,
                      os.path.join(input_dir, "graded.png"))
    plot_correlations(binary_corr,
                      os.path.join(input_dir, "binary.png"))
