from fqi_functions import runFQI
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np
import itertools
from plot_results import plot_qdist
import os
import pickle

def job(lr, iters):
    runFQI('../pickles/fpreppedSamples_04172022.pkl', 'nn', iters=iters, lr=lr, mlp_type='attn')

# take qdiffs from csv file and plot qdiffs with the selected clf and iters
# qdiffs_blank.csv is the template csv file that the csvwriter can write to
def parse_qdiffs_csv(clf, iters, path = 'qdiffs.csv'):
    qdiffs = pd.read_csv(path)
    qdiffs = qdiffs[qdiffs.clf == clf]
    qdiffs = qdiffs[qdiffs.iters == iters]
    qdiffs = qdiffs.groupby(['lr'])

    # make arrays for each learning rate
    groups = qdiffs['qdiff'].apply(np.array)
    lrs = list(groups.index)
    labels = [clf + ' (lr = ' + str(lr) + ')' for lr in lrs]
    Q_dists = list(groups)

    plot_qdist(Q_dists, labels)


def plot_compare_qdiffs(lr_list=None, iters_list=None, model_list=None, model_output_dir="../model_outputs", ylims=None):
    """

    Args:
        lr_list: List of lr's to plot simultaneously
        iters_list: List of number of iters to plot simultaneously
        model_list: "tree", "nn_attn", or "nn_vanilla"
        model_output_dir: path to pkl files

        LEAVE ANYTHING AS NONE TO SELECT ALL VALUES

    """
    Qdists = []
    labels = []

    # These list components are parseable from the files in model_output_dir but
    # I'm hardcoding these for now
    lr_list = lr_list if lr_list else ["0.1", "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07"]
    iters_list = iters_list if iters_list else [30, 50, 100, 200, 300, 400, 500]
    model_list = model_list if model_list else ["tree", "nn_vanilla", "nn_attn"]

    for lr, iters, model in itertools.product(lr_list, iters_list, model_list):
        path = os.path.join(
                model_output_dir,
                f"fqi_{model}_{iters}iters_{lr}lr_04_20_22.pkl") # date(s) are parseable from filenames
        with open(path, 'rb') as f:
            vars = pickle.load(f)
            if model != "tree":
                qdist = vars[2]
            else:
                qdist = vars[4]
            Qdists.append(qdist)
            labels.append(f"model = {model}, lr = {lr}, iters = {iters}")

    plot_qdist(Qdists, labels)


num_cores = multiprocessing.cpu_count()
print(f"num_cores: {num_cores}")
#this was num I actually used when running. Was pretty fast even with this small #
num_cores=4

if __name__== '__main__':
    vals = itertools.product(
            [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], # lr
            [30, 50, 100, 200, 300, 400, 500]) # iters
    Parallel(n_jobs=num_cores)(delayed(job)(lr, iters) for lr, iters in vals)
