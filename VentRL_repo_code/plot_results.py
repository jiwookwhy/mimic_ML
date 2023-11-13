"""
Simple script to construct training plots,
feature importance bar plot (for tree FQI),
violin plots
and attention layer relates visualizations.
"""

# To Do's
# add plotting function looking into Attn MLP layer weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import operator as o
import pickle
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from fqi_clfs import MLPTorch, AttnMLPTorch
from heapq import nlargest

feats = ['Admittype', 'Ethnicity', 'Gender', 'Age', 'Admission Weight (Kg)', 'Height (cm)',
         'Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry', 'Non Invasive Blood Pressure mean',
         'Non Invasive Blood Pressure systolic', 'Non Invasive Blood Pressure diastolic', 'Inspired O2 Fraction',
         'PEEP set', 'Mean Airway Pressure', 'Ventilator Mode', 'Tidal Volume (observed)', 'PH (Arterial)',
         'Respiratory Rate (spontaneous)', 'Richmond-RAS Scale', 'Peak Insp. Pressure', 'O2 Flow',
         'Plateau Pressure', 'Arterial O2 pressure', 'Arterial CO2 Pressure', 'Propofol', 'Fentanyl (Concentrate)',
         'Midazolam (Versed)', 'Fentanyl', 'Dexmedetomidine (Precedex)', 'Morphine Sulfate', 'Hydromorphone (Dilaudid)',
         'Lorazepam (Ativan)', 'timeIn', 'IntNum']


# define some helpers

def plot_qdist(Q_dists, labels):
    """
    Constructs training plots.
    ---
    Args:
    Q_dists: list containing Q_dist arrays (saved by fqi module).
    Should have length >=1.
    labels: list containing labels referring to each Q_dist passed.
    These will make up legends for plot.
    """
    palette = itertools.cycle(sns.color_palette("husl", 9))
    for i in range(len(Q_dists)):
        # get vals to plot
        Q_dist_yvals = [j.item() for j in Q_dists[i]]
        Q_dist_xvals = range(len(Q_dist_yvals))
        # plot them
        plt.plot(Q_dist_xvals, Q_dist_yvals, c=next(palette), label=labels[i])
        plt.xlabel('FQI iterations', fontsize=14)
        plt.ylabel('$Q_{diff} = Q_n - Q_{n-1}$', fontsize=14)
        plt.title('FQI training', fontsize=16)
        plt.legend(loc='best')
    # save final figure
    out_dir = '../figs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'FQI_training.png'))
    plt.clf()


def getImportances(policy_clf, feats):
    """
    Gets importance weights for features in tree policy classifier.
    Args
    ---
    policy_clf: a classifier mapping states --> actions. Needs to have a
    .feature_importances_ attribute!
    feats: list of features used to train policy_clf (in order they were
    parsed).
    """
    weights = policy_clf.feature_importances_
    importanceDf = pd.DataFrame(data={"feats": feats, "importances": weights},
                                index=range(len(feats)))
    return weights, importanceDf


def construct_barplot(axes, data):
    """
    Construct bar plot showing importance weights
    of different features for a tree clf.
    """
    def _barplot(ax, dpoints):
        palette = itertools.cycle(
            [sns.color_palette('Set2')[0], sns.color_palette('Set2')[4]])
        # Aggregate the conditions and the categories according to their mean values
        conditions = [(c, np.mean(dpoints[dpoints[:, 0] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 0])]
        categories = [(c, np.mean(dpoints[dpoints[:, 1] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 1])]
        # sort the conditions, categories and data so that the bars in the plot will be ordered by
        #category and condition
        conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
        categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
        dpoints = np.array(
            sorted(dpoints, key=lambda x: categories.index(x[1])))
        # the space between each set of bars
        space = 0.3
        n = len(conditions)
        width = 0.5  # 0.3
        height = (1 - space) / n
        # Create a set of bars at each position
        for i, cond in enumerate(conditions):
            indices = range(1, len(categories)+1)
            vals = dpoints[dpoints[:, 0] == cond][:, 2].astype('float')
            pos = [j - (1 - space) / 2. + i * height for j in indices]
            ax.barh(pos, vals, height=width, label=cond, color=next(palette))
        ax.set_yticks(indices)
        ax.set_yticklabels(categories, size=12)
        ax.set_xlabel("Feature Importance", size=12)
        ax.set_ylabel("", size=12)
        ax.set_ylim([0, len(categories)+1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower right', fontsize=12)

    _barplot(axes, data)
    out_dir = '../figs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'Importance_Barplot.png'))
    plt.clf()


def get_discretizedActions(samples):
    """
    Takes actions space in raw samples
    and converts these to discretizedActionSum pair
    containing vent decision at idx[0] and dose decision at idx[1].
    """
    for i in range(len(samples)):
        curr_hadm = samples[i]['hadm']
        actions = samples[i]['actions']
        # define discretization scheme (this is same as in main code)
        a1 = (1*(np.logical_and(np.transpose(actions)[1] > 0, np.transpose(actions)[1] <= 2)) +
              2*(np.transpose(actions)[1] > 2))
        a2 = (1*(np.logical_and(np.transpose(actions)[2] > 0, np.transpose(actions)[2] <= 10)) +
              2*(np.transpose(actions)[2] > 10))
        a3 = (1*(np.logical_and(np.transpose(actions)[3] > 0, np.transpose(actions)[3] <= 200)) +
              2*(np.transpose(actions)[3] > 200))
        a4 = (1*(np.logical_and(np.transpose(actions)[4] > 0, np.transpose(actions)[4] <= 10)) +
              2*(np.transpose(actions)[4] > 10))
        a4 = (1*(np.logical_and(np.transpose(actions)[4] > 0, np.transpose(actions)[4] <= 25)) +
              2*(np.transpose(actions)[4] > 25))
        a5 = (1*(np.logical_and(np.transpose(actions)[5] > 0, np.transpose(actions)[5] <= 100)) +
              2*(np.transpose(actions)[5] > 100))
        a6 = (1*(np.logical_and(np.transpose(actions)[6] > 0, np.transpose(actions)[6] <= 2)) +
              2*(np.transpose(actions)[6] > 2))
        a7 = (1*(np.logical_and(np.transpose(actions)[7] > 0, np.transpose(actions)[7] <= 1)) +
              2*(np.transpose(actions)[7] > 1))
        a8 = (1*(np.logical_and(np.transpose(actions)[8] > 0, np.transpose(actions)[8] <= 1)) +
              2*(np.transpose(actions)[8] > 1))
        discretizedActions = np.transpose(
            [np.transpose(actions)[0], a1, a2, a3, a4, a5, a6, a7, a8])
        discretizedActionSum = np.transpose(
            [np.transpose(actions)[0], a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8])
        tmp = np.transpose([np.transpose(discretizedActionSum)[0],
                            np.ceil(np.transpose(discretizedActionSum)[1]/3)])
        discretizedActionSum = tmp
        samples[i]['discretizedActions'] = discretizedActions
        samples[i]['discretizedActionSum'] = discretizedActionSum
    return samples


def get_policy_actions(samples, policy_clf, clf_type):
    """
    Gets action predictions made by our policy clf.
    Assumes clf has a .predict method implemented!
    """
    for i in range(len(samples)):
        #currStates = pd.DataFrame(np.concatenate(samples[i]['currStates'])).to_numpy()
        currStates = np.concatenate(samples[i]['currStates'])
        policy_actions = policy_clf.predict(currStates)
        if clf_type == 'nn':
            # feed through Softmax and take max to get idx of action pair
            policy_actions = F.softmax(
                torch.FloatTensor(policy_actions), dim=1)
            policy_actions = policy_actions.numpy().max(axis=1).astype(int)
        samples[i]['clf_policy_actions'] = policy_actions
    return samples


def get_policy_diff(samples, actionChoices, action_name):
    """
    Get percent match between hospital policy and
    model policy.
    ----
    Args:
    samples: set with all samples. Should already include discretizedActionSum
    and policy predictions.
    actionChoices: list containing 2-long array corresponding to different
    combinations of vent choices (0-1) and dose choices (1 through 4).
    action_name: str. Can be wither 'vent' if constructing difference for vent
    action choies or 'sed' if for sedatives.
    """
    if action_name == 'vent':
        action_idx = 0
    elif action_name == 'sed':
        action_idx = 1
    else:
        raise NotImplementedError('We only take vent or sed as action names!')

    action_col_name = 'percent_match_' + action_name

    for i in range(len(samples)):
        # hosp action choice
        h_d_actionsums = samples[i]['discretizedActionSum']
        h_action = np.array([j[action_idx] for j in h_d_actionsums])

        # model action choice
        m_d_actionsums = np.array([actionChoices[j]
                                  for j in samples[i]['clf_policy_actions']])
        m_action = np.array([j[action_idx] for j in m_d_actionsums])

        # get matching fraction
        # and add it to dict
        percent_match = 100 * \
            (len(np.where(h_action - m_action == 0)[0])/len(h_action))
        samples[i][action_col_name] = percent_match
    return samples


def plot_percent_match_hist(samples, clf_type):
    vent_col_name = 'percent_match_vent'
    seds_col_name = 'percent_match_sed'
    vent_matches = []
    sed_matches = []
    for i in range(len(samples)):
        vent_matches.append(samples[i][vent_col_name])
        sed_matches.append(samples[i][seds_col_name])

    out_dir = '../figs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # plot vent %match histogram
    sns.histplot(np.array(vent_matches), bins=10, color='blue')
    plt.title('% Match between hospital and model policies (vent)', fontsize=14)
    venthist_outfile = clf_type + '_percent_match_hist_vent.png'
    plt.savefig(os.path.join(out_dir, venthist_outfile))
    plt.clf()

    # plot seds %match histogram
    sns.histplot(np.array(sed_matches), bins=10, color='blue')
    plt.title('% Match between hospital and model policies (seds)', fontsize=14)
    sedhist_outfile = clf_type + '_percent_match_hist_seds.png'
    plt.savefig(os.path.join(out_dir, sedhist_outfile))
    plt.clf()
    return


def compute_mean_reward(samples):
    """
    Computes mean discounted rewards for samples.
    This is useful when constructing violin plots.
    """
    for i in range(len(samples)):
        discounts = np.array(
            [0.99**j for j in range(len(samples[i]['rewards']))])
        mean_reward = np.mean(samples[i]['rewards']*discounts)
        samples[i]['mean_reward'] = mean_reward
    return samples


def compute_max_IntNum(samples):
    """
    Gets  max intubatin number
    for each hadm in samples
    """
    for i in range(len(samples)):
        currStates = samples[i]['currStates']
        int_nums = []
        for j in currStates:
            curr_IntNum = j[0][34]
            int_nums.append(curr_IntNum)
        max_IntNum = np.max(np.array(int_nums))
        samples[i]['max_IntNum'] = max_IntNum
    return samples


def get_violin_plots(samples, action_name, clf_type):
    """
    Constructs violin plots for a given action  (vent or seds).
    Each action should yield 2 plots, one looking into mean discounted rewards
    and another into intubation number over hadm.
    ----
    Args:
    samples: dict containing lots of the info compute with functionality above.
    action_name: str. Should be either "vent" or "sed"
    """

    # def base function to construct violin plot
    def _build_violin_plot(data, action, xvars, out_dir, clf_name):
        plt.style.use('seaborn-white')
        pos = [1, 2, 3, 4, 5, 6]
        fig = plt.subplot()
        violin_parts = plt.violinplot(data, pos, vert=False, widths=0.9, showmeans=True,
                                      showextrema=True, showmedians=False)
        with warnings.catch_warnings():
            # ignore userwarning regarding FixedLocator -- this is ok here
            warnings.simplefilter("ignore")
            fig.set_yticklabels(['', '$\Delta_0$', '$\Delta_1$', '$\Delta_2$', '$\Delta_3$',
                                 '$\Delta_4$', '$\Delta_5$'], size=14)
        fig.set_xlabel('{} over admission'.format(xvars), size=14)
        fig.set_ylabel('$\Delta(\pi_{FQI}, \pi_{Hosp})$', size=14)
        fig.set_title('{} policy ({})'.format(action, xvars))

        colour0 = sns.color_palette('Set2')[0]
        colour1 = sns.color_palette('Set2')[1]

        for pc in violin_parts['bodies']:
            pc.set_facecolor(colour0)
            pc.set_edgecolor(colour0)
        for pc in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            violin_parts[pc].set_color(colour0)

        out_fname = clf_name + '_violin_{}_{}'.format(action, xvars)
        plt.savefig(os.path.join(out_dir, out_fname))
        plt.clf()
        return

    # now prep date to construct plots for a given action (i.e., vent or seds)
    # mk output dir if it doesn't exist
    out_dir = '../figs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if action_name not in ['vent', 'sed']:
        raise NotImplementedError(
            "Only 'vent' and 'sed' are implemented actions!")

    percent_match = []
    mean_reward = []
    max_IntNum = []
    action_match_col_name = 'percent_match_' + action_name
    # Get match % for action. mean_reward and #intubations
    for i in range(len(samples)):
        percent_match.append(samples[i][action_match_col_name])
        mean_reward.append(samples[i]['mean_reward'])
        max_IntNum.append(samples[i]['max_IntNum'])
    # construct bins based on % match in policies
    bins = [100, 90, 70, 50, 30, 0]
    dists = [np.where(np.array(percent_match) >= bins[0])[0]]
    for i in range(len(bins)-1):
        dset = np.where(np.logical_and((np.array(percent_match)
                        >= bins[i+1]), (np.array(percent_match) < bins[i])))[0]
        dists.append(dset)
    # now get data we need for violin plot construction
    action_rewards = {}
    action_IntNum = {}
    nd = 0
    for d in dists:
        action_rewards[nd] = np.array(mean_reward)[d]
        action_IntNum[nd] = np.array(max_IntNum)[d]
        nd = nd + 1
    reward_data = [action_rewards[i] for i in range(0, 6)]
    intNum_data = [action_IntNum[i] for i in range(0, 6)]
    # finally contruct plots for action!
    # for reward
    _build_violin_plot(reward_data, action_name,
                       'mean_reward', out_dir, clf_type)
    _build_violin_plot(intNum_data, action_name,
                       'intubation_number', out_dir, clf_type)
    return


def plot_attention_heatmap(hadm_data, action_choices, policy_clf, plt_name="heatmap.png"):
    """

    Args:
        hadm_data: Specific element from list: all_samples_na
        action_choices: should be actionChoices_na
        policy_clf: should be Q_policy_na
    """
    # since my brain is tired I'm hardcoding the sample
    # also assumes you pass an attention classifier, not vanilla_mlp or tree

    # https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    # `activation` will store the attention weights
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # should activate hook on call to
    clf_handle = policy_clf.fc_layers[0].register_forward_hook(
        get_activation("attn_layer"))

    # predict and collect attention weights
    currStates = np.concatenate(hadm_data['currStates'])
    policy_actions = policy_clf.predict(currStates)
    attn_weights = F.softmax(activation['attn_layer'].T, dim=0)

    # deactive hook (so don't keep adding to hook dict)
    clf_handle.remove()

    decisions = []
    vent = 0
    seds = 1
    for action_idx in [vent, seds]:
        h_d_actionsums = hadm_data['discretizedActionSum']
        h_action = np.array([j[action_idx] for j in h_d_actionsums])
        decisions.append(h_action)

        # model action choice
        m_d_actionsums = np.array([action_choices[j]
                                  for j in hadm_data['clf_policy_actions']])
        m_action = np.array([j[action_idx] for j in m_d_actionsums])
        decisions.append(m_action)

    decisions = np.vstack(decisions)
    # create heatmap
    plt.rcParams["figure.figsize"] = (12, 10)
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, gridspec_kw={'height_ratios': [3, 0.3, 0.3]})
    im1 = ax1.imshow(attn_weights, aspect='auto')
    ax1.set_yticks(range(len(feats)))
    ax1.set_yticklabels(feats)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Attention Weights", rotation=0, labelpad=50)

    vent_decisions = decisions[0:2]
    im2 = ax2.imshow(vent_decisions, aspect='auto',
                     cmap=plt.cm.get_cmap('viridis', 2))
    ax2.set_yticks(range(vent_decisions.shape[0]))
    ax2.set_yticklabels(["Hospital Vent", "Model Vent"])
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0.25, 0.75], aspect=2)
    cbar2.ax.set_yticklabels(["off", "on"])
    cbar2.set_label("Ventilator Status", rotation=0, labelpad=50)

    seds_decisions = decisions[2:]
    im3 = ax3.imshow(seds_decisions, aspect='auto',
                     cmap=plt.cm.get_cmap('viridis', 4))
    im3.set_clim(0, 4)
    ax3.set_yticks(range(seds_decisions.shape[0]))
    ax3.set_yticklabels(["Hospital Sed", "Model Sed"])
    plt.xlabel("Hours Since Admission")
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0.5, 1.5, 2.5, 3.5], aspect=2)
    cbar3.ax.set_yticklabels(["1", "2", "3", "4"])
    cbar3.set_label("Sedation Levels", rotation=0, labelpad=50)

    out_dir = '../figs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, plt_name), bbox_inches="tight")
    plt.clf()


def get_low_high_match(samples, k, action_name, method="highest", min_stay_length=10, max_stay_length=200):
    """

    Args:
        samples: list containing hadm data
        k: number of sample indices to return (NOT HADM_IDS)
        action_name: "vent" or "sed" or "both"
        method: "highest" or "lowest"
    """

    if action_name == 'vent':
        action_idx = 0
    elif action_name == 'sed':
        action_idx = 1
    elif action_name == 'both':
        pass
    else:
        raise NotImplementedError('We only take vent or sed as action names!')

    percent_matchs = {}
    for i, sample in enumerate(samples):
        hours_in_stay = len(sample['currStates'])
        if hours_in_stay < min_stay_length or hours_in_stay > max_stay_length:
            continue
        if action_name == "both":
            percent_matchs[i] = (sample["percent_match_vent"]) / 2
            percent_matchs[i] += (sample["percent_match_sed"]) / 2
        else:
            percent_matchs[i] = (sample["percent_match_" + action_name])

    reversed = True if method == "highest" else False
    return sorted(percent_matchs, key=lambda x: percent_matchs[x], reverse=reversed)[:k]

# now let's load some data and use the above to get all of our figs


def main():

    #########################
    # Read in tree model info
    ########################
    tree_100iters = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/pickles/fqi_tree_100iters_04_18_22.pkl'
    with open(tree_100iters, 'rb') as f:
        tree_res = pickle.load(f)

    Qest_t, Q_policy_t, Q_t, Qold_t, Qdist_t, S_t, optA_t, actionChoices_t = tree_res[0], tree_res[1], tree_res[2], tree_res[3], \
        tree_res[4], tree_res[5], tree_res[6], tree_res[7]

    #########################
    # Read in vanilla MLP info
    #########################
    mlp_vanilla_arr = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/model_outputs_cv_vanilla_MLP/fqi_nn_vanilla_100iters_1e-07lr_04_20_22.pkl'
    mlp_vanilla_policy_clf = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/model_outputs_cv_vanilla_MLP/fqi_nn_vanilla_100iters_1e-07lr_04_20_22_clf.pth'
    with open(mlp_vanilla_arr, 'rb') as f2:
        mlp_vanilla_res = pickle.load(f2)
    Q_nv, Qold_nv, Qdist_nv, S_nv, optA_nv, actionChoices_nv = mlp_vanilla_res[0], mlp_vanilla_res[1], mlp_vanilla_res[2], \
        mlp_vanilla_res[3], mlp_vanilla_res[4], mlp_vanilla_res[5]

    # get vanilla mlp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_vanilla_sd = torch.load(mlp_vanilla_policy_clf)
    Q_policy_nv = MLPTorch(35, 9, (5, 2)).to(device)
    Q_policy_nv.load_state_dict(mlp_vanilla_sd)

    #######################
    # Red in attn MLP info
    #######################
    mlp_attn_arr = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/model_outputs_cv_attn_MLP/fqi_nn_attn_100iters_1e-07lr_04_20_22.pkl'
    mlp_attn_policy_clf = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/model_outputs_cv_attn_MLP/fqi_nn_attn_100iters_1e-07lr_04_20_22_clf.pth'
    with open(mlp_attn_arr, 'rb') as f3:
        mlp_attn_res = pickle.load(f3)
    Q_na, Qold_na, Qdist_na, S_na, optA_na, actionChoices_na = mlp_attn_res[0], mlp_attn_res[1], mlp_attn_res[2], \
        mlp_attn_res[3], mlp_attn_res[4], mlp_attn_res[5]

    # get attn mlp
    mlp_attn_sd = torch.load(mlp_attn_policy_clf)
    Q_policy_na = AttnMLPTorch(35, 9, (5, 2)).to(device)
    Q_policy_na.load_state_dict(mlp_attn_sd)

    #####################
    #Read in samples
    ####################
    samples_file = '/home/daniela/ECE_Courses/STAT561/Final_Project/VentRL-master/pickles/allSamples_04172022.pkl'
    with open(samples_file, 'rb') as f4:
        all_samples = pickle.load(f4)

    #######################
    # Get training plots
    ########################
    plot_qdist([Qdist_t, Qdist_nv, Qdist_na], [
               'tree', 'MLP vanilla', 'MLP attention'])

    #####################
    # plot feature importance
    # for tree only
    #####################
    feats = ['Admittype', 'Ethnicity', 'Gender', 'Age', 'Weight (Kg)', 'Height (cm)',
             'Heart Rate', 'Respiratory Rate', '$SpO_2$', 'Mean BP',
             'Systolic BP', 'Diastolic BP', 'Inspired O2 Fraction',
             'PEEP set', 'Mean Airway Pressure', 'Ventilator Mode', 'Tidal Volume', 'PH (Arterial)',
             'Resp. Rate(spont.)', 'Richmond-RAS Scale', 'Peak Insp. Pressure', 'O2 Flow',
             'Plateau Pressure', 'Arterial O2 pressure', 'Arterial CO2 Pressure', 'Propofol', 'Fentanyl (Conct.)',
             'Midazolam', 'Fentanyl', 'Dexmedetomidine', 'Morphine Sulfate', 'Hydromorphone',
             'Lorazepam', 'timeIn', 'IntNum']
    t_weights, t_importanceDf = getImportances(Q_policy_t, feats)
    fig = plt.figure()
    ax = plt.subplot()
    fig.set_size_inches(10, 8)
    dpoints = np.transpose(
        np.vstack([['Neural Fitted-Q']*len(feats), feats, t_weights]))
    construct_barplot(ax, dpoints)
    plt.clf()

    ###############################
    # Plot policy % match histograms
    ################################

    # for tree
    all_samples_t = get_discretizedActions(all_samples)
    all_samples_t = get_policy_actions(all_samples_t, Q_policy_t, 'tree')
    actionChoices_list_t = actionChoices_t.tolist()
    all_samples_t = get_policy_diff(
        all_samples_t, actionChoices_list_t, 'vent')
    all_samples_t = get_policy_diff(all_samples_t, actionChoices_list_t, 'sed')
    plot_percent_match_hist(all_samples_t, 'tree')

    # for vanilla mlp
    all_samples_nv = get_discretizedActions(all_samples)
    all_samples_nv = get_policy_actions(all_samples_nv, Q_policy_nv, 'nn')
    actionChoices_list_nv = actionChoices_nv.tolist()
    all_samples_nv = get_policy_diff(
        all_samples_nv, actionChoices_list_nv, 'vent')
    all_samples_nv = get_policy_diff(
        all_samples_nv, actionChoices_list_nv, 'sed')
    plot_percent_match_hist(all_samples_nv, 'vanilla_mlp')

    # for attn mlp
    all_samples_na = get_discretizedActions(all_samples)
    all_samples_na = get_policy_actions(all_samples_na, Q_policy_na, 'nn')
    actionChoices_list_na = actionChoices_na.tolist()
    all_samples_na = get_policy_diff(
        all_samples_na, actionChoices_list_na, 'vent')
    all_samples_na = get_policy_diff(
        all_samples_na, actionChoices_list_na, 'sed')
    plot_percent_match_hist(all_samples_na, 'attn_mlp')

    ######################
    # Get violin plots
    ######################
    # for tree
    all_samples_t = compute_mean_reward(all_samples_t)
    all_samples_t = compute_max_IntNum(all_samples_t)
    get_violin_plots(all_samples_t, 'vent', 'tree')
    get_violin_plots(all_samples_t, 'sed', 'tree')

    # for vanilla mlp
    all_samples_nv = compute_mean_reward(all_samples_nv)
    all_samples_nv = compute_max_IntNum(all_samples_nv)
    get_violin_plots(all_samples_nv, 'vent', 'vanilla_mlp')
    get_violin_plots(all_samples_nv, 'sed', 'vanilla_mlp')

    # for attn mlp
    all_samples_na = compute_mean_reward(all_samples_na)
    all_samples_na = compute_max_IntNum(all_samples_na)
    get_violin_plots(all_samples_na, 'vent', 'attn_mlp')
    get_violin_plots(all_samples_na, 'sed', 'attn_mlp')

    worst_sed_idx = get_low_high_match(
        all_samples_na, 1, "sed", method="lowest")[0]
    best_sed_idx = get_low_high_match(
        all_samples_na, 1, "sed", method="highest")[0]
    worst_vent_idx = get_low_high_match(
        all_samples_na, 1, "vent", method="lowest")[0]
    best_vent_idx = get_low_high_match(
        all_samples_na, 1, "vent", method="highest")[0]
    worst_both_idx = get_low_high_match(
        all_samples_na, 1, "both", method="lowest")[0]
    best_both_idx = get_low_high_match(
        all_samples_na, 1, "both", method="highest")[0]

    plot_attention_heatmap(
        all_samples_na[worst_sed_idx], actionChoices_list_na, Q_policy_na, plt_name="worst_sed_heatmap.png")
    plot_attention_heatmap(
        all_samples_na[best_sed_idx], actionChoices_list_na, Q_policy_na, plt_name="best_sed_heatmap.png")
    plot_attention_heatmap(
        all_samples_na[worst_vent_idx], actionChoices_list_na, Q_policy_na, plt_name="worst_vent_heatmap.png")
    plot_attention_heatmap(
        all_samples_na[best_vent_idx], actionChoices_list_na, Q_policy_na, plt_name="best_vent_heatmap.png")
    plot_attention_heatmap(
        all_samples_na[worst_both_idx], actionChoices_list_na, Q_policy_na, plt_name="worst_both_heatmap.png")
    plot_attention_heatmap(
        all_samples_na[best_both_idx], actionChoices_list_na, Q_policy_na, plt_name="best_both_heatmap.png")


if __name__ == "__main__":
    main()
