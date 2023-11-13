# Here we actually use gp_module to:
# 1) train a gp for each ID and feature being evaluated
# 2) get predicitons/samples for time pts we want to access (considering a given sampling rate)
# 3) write all of these into a single file containing gp_times and their corresponding measurements (as predicted)
# for each feature, group by ID and in asceding time order.
# Once we get outputs from this script containing info/predictions for all features we want and all IDs
# then we can feed this into a loader and into our model directly (with only simple normalization as last preprocessing step).
#NOTE: I clamp outputs I get to force certain physiological constraints in some of the features ...
#This might or not be case for other features
#So CHECK IT OUT AND ADJUST ACCORDINGLY

import pandas as pd
import numpy as np
import torch
import gpytorch
from gp_module import ExactGPModel, train_GP, get_samples
from tqdm import tqdm
import os

#def some helpers
def setup_gp_timings(df, icu_id, feat_key):
    """
    Args
    ---
    df: clean df (read with pandas )

    Returns
    ---
    Essentially df with what we want to feed into GP (for a given ID)
    and a version of it containing full gp_times series... This is useful for
    sampling all features within same range.
    """
    #get all df entries for id
    keys = ['icustay_id', 'charttime']
    keys.append(feat_key)
    id_entries = df.loc[df['icustay_id']==icu_id, keys]
    #now process charttime
    #convert to datetime
    id_entries['charttime'] = pd.to_datetime(id_entries['charttime'])
    #sort by charttime
    id_entries = id_entries.sort_values(by='charttime')
    #now get gp input times
    gp_times = ((id_entries['charttime']).diff())/np.timedelta64(1, 'm')
    gp_times = gp_times.cumsum()
    #replace first entry (which is NaN by default...) with a zero
    #this is ONLY NaN in this Series, by design
    gp_times = gp_times.replace(to_replace=np.nan, value=0)
    #add this series to our processed df
    id_entries['gp_times'] = gp_times
    #drop entries with NaN in feature val
    c_id_entries = id_entries.dropna(axis=0)
    #drop any random duplicates lingering around
    dup_keys = ['gp_times'].append(feat_key)
    c_id_entries = c_id_entries.drop_duplicates(subset=dup_keys)
    return id_entries, c_id_entries

#read cleaned up file you want to process
clean_vent_path = '../../clean_forGP_data/vent_settings.csv'
clean_vent = pd.read_csv(clean_vent_path, index_col=0)

#get cohort IDs && features
#again am loading cohort straight from file b/c for vent case this is ok
#in general, this should come from cohort_info file
cohort = pd.unique(clean_vent['icustay_id'])
features = list(clean_vent.columns)
non_feat_cols = ['Index', 'subject_id', 'hadm_id', \
'icustay_id', 'charttime', 'rass']
features = [i for i in features if i not in non_feat_cols]

#loop through IDs
all_gp_estimates = []

for c in tqdm(range(len(cohort))):
    icu_id = int(cohort[c])
    for f in range(len(features)):
        feat = features[f]
        #get inputs ready
        gp_df, c_gp_df = setup_gp_timings(clean_vent, icu_id, feat)
        gp_train_x = torch.from_numpy(c_gp_df['gp_times'].to_numpy()).float().cuda()
        gp_train_y = torch.from_numpy(c_gp_df[feat].to_numpy()).float().cuda()
        #create model && likelihood instances
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        model = ExactGPModel(gp_train_x, gp_train_y, likelihood).cuda()
        #train model
        model = train_GP(model, likelihood, gp_train_x, gp_train_y, icu_id, feat)
        # get predictions
        pred = get_samples(model, likelihood, gp_df, feat)
        if f ==0:
            #create starting df for ID
            id_df = pd.DataFrame(pred)
        else:
            #simply append predictions for new feature to existing ID df
            id_df[feat] = pred[feat]
    all_gp_estimates.append(id_df)

#concatenate all ID dfs to get our final result/output
all_gp_estimates = pd.concat(all_gp_estimates, axis=0, ignore_index=True)
#write it out
out_dir = '../../gp_outputs'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
all_gp_estimates.to_csv(os.path.join(out_dir, 'vent_settings_gp_out_04132022_test.csv'))
