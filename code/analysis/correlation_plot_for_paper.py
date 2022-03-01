import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import xarray as xr

def get_prediction(path):
    pred = pd.read_csv(path, sep="\t", header=None, index_col=0)
    if not len(pred.columns) == 1:
        return
    pred.columns = [os.path.basename(path).replace(".tsv", "")]
    return pred

def get_correlations(predictions, lang, method):
    predictions = [get_prediction(os.path.join(input_dir, lang, p)) for p in predictions]
    predictions = pd.concat(predictions, axis=1)
    corr = predictions.corr(method=method)
    return corr


if __name__ == '__main__':
    input_dir = sys.argv[1]
    languages = ["english", "german", "italian", "latin", "norwegian1", "norwegian2", "russian1", "russian2", "russian3", "swedish"]

    graded_corr = {}
    binary_corr = {}
    
    for lang in languages:
        print(lang)
        predictions = sorted([p for p in os.listdir(os.path.join(input_dir, lang)) if p.endswith('tsv')])
        
        graded = [p for p in predictions if p in ['SGNS_raw.tsv',
                                                  'apd.tsv',
                                                  'cos.tsv',
                                                  'jsd.tsv',
                                                  'profile_morph_5_graded.tsv',
                                                  'profile_synt_5_graded.tsv']]
                  

        binary = [p for p in predictions if p in ['SGNS_raw_binary.tsv',
                                                  'apd_binary.tsv',
                                                  'cos_binary.tsv',
                                                  'jsd_binary.tsv',
                                                  'profile_morph_5_binary.tsv',
                                                  'profile_synt_5_binary.tsv']]


        
        graded_corr[lang] = get_correlations(graded, lang, method = 'spearman')
        binary_corr[lang] = get_correlations(binary, lang, method = 'pearson')
        
    graded_array = xr.DataArray([graded_corr[lang] for lang in languages])
    graded_mean = graded_array.mean(axis=0)

    plt.figure()
    plot = sns.heatmap(graded_mean, 
                       xticklabels=["SGNS", "APD", "PRT", "JSD", "MORPH", "SYNT"],
                       yticklabels=["SGNS", "APD", "PRT", "JSD", "MORPH", "SYNT"],
                       cmap="BuPu")
    plot.figure.tight_layout()
    plt.savefig("graded.png")

