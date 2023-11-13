import pandas as pd
import time
import numpy as np
import pickle
import joblib
from joblib import Parallel, delayed
import os
import csv
from pandas import *
import datetime as dt
from datetime import date, datetime, timedelta
from collections import Counter
import math
import copy as cp
from fqi_clfs import fitTree, fitCtree, fitTNN
import torch
from torch import nn
import torch.nn.functional as F

#def main function to run FQI/NFQI
def runFQI(samples_file, clf, mlp_type=None, iters= 100, batchSize=50000, train_test_split=0.90, gamma=0.9, lr=1e-5):
    """
    Should run same steps as in fqi.py script, only with more flexibility....
    Admits different types of clf and choices for gamma, train_test_split,
    batchsize and iterations.
    ----
    Args:
    samples_file: pickled file containing output from filter.py script.
    clf: can be either 'tree' or 'nn'. Can also adapt this to any other clf obj.
    iters: number of Q-iterations to run.
    batchSize: size of random batch to take from trainingSet during each Q-iteration.
    Needs to be <= actual trainingSet (samples w/out replacement).
    train_test_split: ratio between train and test sets to use. Default (0.90) splits data
    using ~90% of samples for training. There is some rounding here...
    Gamma: weight for descounting rewards...
    """
    with open(samples_file,'rb') as f:
        currStatesC, nextStatesC, actions, rewards, discretizedActions, discretizedActionSum = pickle.load(f)

    # Set up train and test sets

    # Get indices for von/voff time pts
    voffIndices = np.where(np.transpose(currStatesC)[33] == 0)[0]
    vonIndices = np.nonzero(np.transpose(currStatesC)[33])[0]

    #split these according to train/test split
    train_voff_idxs = int(np.rint(train_test_split*voffIndices.shape[0]) + 1)
    train_von_idxs = int(np.rint(train_test_split*vonIndices.shape[0]) + 1)

    #now use these to get train and test sets
    trainingSet = np.sort(np.concatenate((voffIndices[:train_voff_idxs], vonIndices[:train_von_idxs]) , axis=0))
    testSet = np.sort(np.concatenate((voffIndices[train_voff_idxs:], vonIndices[train_von_idxs:]) , axis=0))

    #prep samples and action choices
    samples = np.hstack((currStatesC, discretizedActionSum))
    actionChoices = (np.vstack({tuple(row) for row in discretizedActionSum})) #this might raise FutureWarning... yup it does! Will clean it up!

    #check clf type and if 'nn', make sure mlp_type is also setup correctly
    if clf=='nn':
        assert(mlp_type!=None and mlp_type in ('vanilla', 'attn')), \
        "When clf=='nn', mlp_type needs to be either 'vanilla' or 'attn'"

    #init model
    print('Initialization')
    if clf=='tree':
        Qest = fitTree(samples[trainingSet], rewards[trainingSet])
    elif clf=='nn':
        if mlp_type=='vanilla':
            Qest = fitTNN(samples[trainingSet], rewards[trainingSet], "continuous", mlp_type='vanilla', lr=1e-5)
        else:
            Qest = fitTNN(samples[trainingSet], rewards[trainingSet], "continuous", mlp_type='attn', lr=1e-5)
    else:
        raise NotImplementedError("Chosen clf has not been implemented! Choose between 'nn' or 'tree'")

    Q = np.zeros((len(actionChoices), len(trainingSet)))
    Qdist = []

    #now do Q-iter procedure
    print('Q-iteration')
    iter = 0
    while iter < iters:
        batch_idx = np.random.choice(len(trainingSet), batchSize, replace=False)
        batch = trainingSet[batch_idx]

        S = {}
        Qold = cp.deepcopy(Q)
        anum = 0
        for a in actionChoices:
            X = np.hstack((nextStatesC[batch], np.ones((batchSize, 1)) * a))
            Q[anum,batch_idx] = Qest.predict(X).reshape(-1)
            anum += 1
        Qdist.append(np.array(np.mean(abs(np.matrix(Qold) - np.matrix(Q)))))
        optA = list((np.argmax(Q, axis=0))[batch_idx])
        T = rewards[batch] + gamma * np.max(Q, axis=0)[batch_idx]

        if clf=='tree':
            Qest = fitTree(samples[batch], T)
        elif clf=='nn':
            if mlp_type=='vanilla':
                Qest = fitTNN(samples[batch], T, "continuous", model=Qest, epochs=1, mlp_type='vanilla', lr=lr)
            else:
                Qest = fitTNN(samples[batch], T, "continuous", model=Qest, epochs=1, mlp_type='attn', lr=lr)
        else:
            raise NotImplementedError("Chosen clf has not been implemented! Choose between 'nn' or 'tree'")

        S = {'n': batch, 'T': T, 'optA': optA}
        print('Iter: {}; Qdiff:{}'.format(iter, Qdist[len(Qdist)-1]))
        f = open("qdiffs.csv", "a")
        writer = csv.writer(f)
        writer.writerow([clf, lr, iters, iter, Qdist[len(Qdist)-1]])
        f.close()
        iter = iter + 1

    print('Policy classifier')
    optA = list(np.argmax(Q, axis=0))

    if clf=='tree':
        policy_clf = fitCtree(currStatesC[trainingSet], optA)
    else:
        optA_hot = F.one_hot(torch.LongTensor(optA), num_classes=9)
        if mlp_type=='vanilla':
            policy_clf = fitTNN(currStatesC[trainingSet], optA_hot, "discrete", mlp_type='vanilla', lr=lr)
        else:
            policy_clf = fitTNN(currStatesC[trainingSet], optA_hot, "discrete", mlp_type='attn', lr=lr)

    #setup output file name
    #should contain date, #iters and clf choice
    td = date.today()
    str_date = td.strftime("%m_%d_%y")
    if clf == 'nn':
        fout_name = 'fqi_' + clf + '_' + mlp_type + '_{}iters_'.format(str(iters)) + f"{lr}lr_" + str_date
    else:
        fout_name = 'fqi_' + clf + '_{}iters_'.format(str(iters)) + str_date

    out_dirs = '../model_outputs'
    if not os.path.exists(out_dirs):
        os.makedirs(out_dirs)
    if clf == "tree":
        with open(os.path.join(out_dirs, fout_name + ".pkl"),'wb') as f:
            pickle.dump((Qest, policy_clf, Q, Qold, Qdist, S, optA, actionChoices), f)
    else:
        with open(os.path.join(out_dirs, fout_name + '.pkl'),'wb') as f:
            pickle.dump((Q, Qold, Qdist, S, optA, actionChoices), f)
        torch.save(Qest.state_dict(), os.path.join(out_dirs, fout_name + "_est" + ".pth"))
        torch.save(policy_clf.state_dict(), os.path.join(out_dirs, fout_name + "_clf" + ".pth"))

    return Qest, Q, Qdist, policy_clf

def main():
    #am using 1 iter for testing
    #will amp it up (paper uses 100 Qiterations for larger dset)
    #train it with tree classifier
    print('Running FQI w/ tree clf:')
    Qest, Q, Qdist, policy_clf = runFQI('../pickles/fpreppedSamples_04172022.pkl', 'tree', iters=1)

    #train it with nn clf, vanilla
    print('Running FQI w/ vanilla MLP clf:')
    Qest, Q, Qdist, policy_clf = runFQI('../pickles/fpreppedSamples_04172022.pkl', 'nn', iters=1, mlp_type='vanilla')

    #training it w/ nn clf, attn
    print('Running FQI w/ MLP clf:')
    Qest, Q, Qdist, policy_clf = runFQI('../pickles/fpreppedSamples_04172022.pkl', 'nn', iters=1, mlp_type='attn')

if __name__== '__main__':
    main()
