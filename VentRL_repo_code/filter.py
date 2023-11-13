#%pylab inline
import pandas as pd
import numpy as np
import pickle
import copy as cp
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

with open("../pickles/preppedSamples_04172022.pkl",'rb') as f:
    currStatesC, nextStatesC, actions, rewards, discretizedActions, discretizedActionSum = pickle.load(f)

#ordering here is all wrong!!!
#feats= ['Age', 'Weight', 'Heart Rate', 'Respiratory Rate', '$SpO_2$',
#        'BP (Mean)', 'BP (Systolic)', 'BP (Diastolic)','$FiO_2$', 'PEEP set',
#        'Mean Airway Pressure', 'Ventilator Mode', 'Tidal Volume', 'Arterial pH', 'RR (Spont)',
#        'RASS','Peak Inspiratory Pressure', '$O_2$ Flow', 'Plateau Pressure','Arterial $O_2$ pressure',
#        'Arterial $CO_2$ pressure', 'Fentanyl (Conc)','Midazolam', 'Propofol', 'Fentanyl',
#        'Dexmedetomidine', 'Morphine', 'Hydromorphone', 'Lorazepam','Time on vent',
#        'Intubation number','Admit Type','Ethnicity','Gender']

#here are actual feats we use (in order)...
feats = ['Admittype', 'Ethnicity', 'Gender', 'Age', 'Admission Weight (Kg)', 'Height (cm)',
         'Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry', 'Non Invasive Blood Pressure mean',
         'Non Invasive Blood Pressure systolic', 'Non Invasive Blood Pressure diastolic', 'Inspired O2 Fraction',
         'PEEP set', 'Mean Airway Pressure', 'Ventilator Mode', 'Tidal Volume (observed)', 'PH (Arterial)',
         'Respiratory Rate (spontaneous)', 'Richmond-RAS Scale', 'Peak Insp. Pressure', 'O2 Flow',
         'Plateau Pressure', 'Arterial O2 pressure', 'Arterial CO2 Pressure', 'Propofol', 'Fentanyl (Concentrate)',
         'Midazolam (Versed)', 'Fentanyl', 'Dexmedetomidine (Precedex)', 'Morphine Sulfate', 'Hydromorphone (Dilaudid)',
         'Lorazepam (Ativan)', 'timeIn', 'IntNum']

# Check if vent on based on timeIn != 0
#idx for timeIn is actually 33, but am using name here!
voffIndices = np.where(np.transpose(currStatesC)[feats.index('timeIn')] == 0)[0]
vonIndices = np.nonzero(np.transpose(currStatesC)[feats.index('timeIn')])[0]

#use 90%train/10%test split as in fqi_functions script
train_voff_idxs = int(np.rint(0.9*voffIndices.shape[0]) + 1)
train_von_idxs = int(np.rint(0.9*vonIndices.shape[0]) + 1)

trainingSet = np.sort(np.concatenate((voffIndices[:train_voff_idxs], vonIndices[:train_von_idxs]) , axis=0))
testSet = np.sort(np.concatenate((voffIndices[train_voff_idxs:], vonIndices[train_von_idxs:]) , axis=0))

samples = np.hstack((currStatesC, discretizedActionSum))

testSamples = {}; testStates = {}; testActions = {}; testRewards = {}; testPt = {}
pt = 0; age = 0; wt = 0
for i in testSet:
    if ((samples[i][3] == age) and (samples[i][4] == wt)):
        testSamples[pt].append(samples[i])
        testStates[pt].append(currStatesC[i])
        testActions[pt].append(discretizedActionSum[i])
        testRewards[pt].append(rewards[i])
        testPt[pt].append(i)
    else:
        pt = pt + 1
        testSamples[pt] = [samples[i]]
        testStates[pt] = [currStatesC[i]]
        testActions[pt] = [discretizedActionSum[i]]
        testRewards[pt] = [rewards[i]]
        testPt[pt] = [i]
    age = samples[i][3]; wt = samples[i][4]

trainSamples = {}; trainStates = {}; trainActions = {}; trainRewards = {}; trainPt = {}
pt = 0; age = 0; wt = 0
for i in trainingSet:
    if ((samples[i][3] == age) and (samples[i][4] == wt)):
        trainSamples[pt].append(samples[i])
        trainStates[pt].append(currStatesC[i])
        trainActions[pt].append(discretizedActionSum[i])
        trainRewards[pt].append(rewards[i])
        trainPt[pt].append(i)
    else:
        pt = pt + 1
        trainSamples[pt] = [samples[i]]
        trainStates[pt] = [currStatesC[i]]
        trainActions[pt] = [discretizedActionSum[i]]
        trainRewards[pt] = [rewards[i]]
        trainPt[pt] = [i]
    age = samples[i][3]; wt = samples[i][4]

upper = [np.mean(np.transpose(currStatesC)[i]) + 4*np.std(np.transpose(currStatesC)[i]) for i in range(35)]
lower = [np.mean(np.transpose(currStatesC)[i]) - 3*np.std(np.transpose(currStatesC)[i]) for i in range(35)]

#use = [1, 4, 6, 7, 9, 13, 14, 16, 18] old idxs
use = ['Admission Weight (Kg)', 'O2 saturation pulseoxymetry', 'Non Invasive Blood Pressure systolic', \
'Non Invasive Blood Pressure diastolic', 'PEEP set',  'PH (Arterial)', 'Respiratory Rate (spontaneous)', \
'Peak Insp. Pressure', 'Plateau Pressure']
use_idx = [feats.index(i) for i in use]
inds = np.unique(np.where([[(currStatesC[trainPt[k][0]][i] <= lower[i] or currStatesC[trainPt[k][0]][i] >= upper[i])
                         for i in use_idx] for k in trainPt.keys()])[0])
inds2 = np.unique(np.where([[(currStatesC[testPt[k][0]][i] <= lower[i] or currStatesC[testPt[k][0]][i] >= upper[i])
                          for i in use_idx] for k in testPt.keys()])[0])

a = (np.concatenate([trainPt[x+1] for x in inds]))
b = (np.concatenate([testPt[x+1] for x in inds2]))
c = (np.sort(np.append(a, b)))
mask = np.ones(len(currStatesC),dtype=bool) #np.ones_like(a,dtype=bool)
mask[c] = False
d = np.where(mask)[0]

print('CurrStatesC prior to filtering:{}'.format(len(currStatesC))) #nextStatesC, discretizedActionSum, rewards and actions have same lenth at this stage
fcurrStatesC = np.array([currStatesC[i] for i in d])
fnextStatesC = np.array([nextStatesC[i] for i in d])
fdiscretizedActionSum = np.array([discretizedActionSum[i] for i in d])
frewards = np.array([rewards[i] for i in d])
fdiscretizedActions = np.array([discretizedActions[i] for i in d])
factions = np.array([actions[i] for i in d])
print('CurrStatesC after filtering:{}'.format(len(currStatesC))) #again other sample variables will have same size post filtering!

with open("../pickles/fpreppedSamples_04172022.pkl",'wb') as f:
    pickle.dump((fcurrStatesC, fnextStatesC, factions, frewards, fdiscretizedActions, fdiscretizedActionSum), f)

#currStatesC[264325], [trainPt[k][0] for k in trainPt.keys()][1806]
#len(currStatesC), len(filtered_cs)

for i in range(35):
    print(i, feats[i], len(np.unique(np.transpose(fcurrStatesC)[i])), np.ptp(np.transpose(fcurrStatesC)[i]))

sns.histplot(data = np.unique(np.transpose(currStatesC)[feats.index('IntNum')]), bins=100)
plt.savefig('../figs/sns_distplot_NumIntubations.png')
