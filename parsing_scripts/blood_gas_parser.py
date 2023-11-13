from utils import mimic_parser, get_lastval_interpolation
import pandas as pd

import pandas as pd
import numpy as np


blood_gas = pd.read_csv('mimiciii_blood_gas.csv')
keys = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']
blood_gas_parsed = mimic_parser(keys, blood_gas)
blood_gas_parsed['CHARTTIME'] = pd.to_datetime(blood_gas_parsed.CHARTTIME)
blood_gas_parsed = blood_gas_parsed.sort_values(keys)

# remove certain icustay ids
bad_icustayids = [220948, 267891, 278117, 296540]
blood_gas_parsed = blood_gas_parsed[blood_gas_parsed.ICUSTAY_ID.isin(bad_icustayids) == False]

# check out summary stats for our features (spot check for weirdness)
summary = blood_gas_parsed.describe()

# drop columns with less than 2000 observations
# columns with fewer than 2000 points are likely to cause difficulties down the line
# (2000 is fewer than a quarter of all patients in the dataset)
usable_columns =  ['CHARTTIME'] + list(summary.columns[summary.loc['count', ] > 2000]) 


# from the summary table above we know we can remove intubated, ventilator, ventilationrate 
# also know no fio2 is stored as a percentage in this table 
# run basic filters on fio2 and peep using the same filtering scheme Daniela implemented for vitals
blood_gas_parsed = blood_gas_parsed.drop(axis = 1, labels = ['ventilator', 'intubated', 'ventilationrate'])
blood_gas_parsed['fio2'] = np.where((blood_gas_parsed['fio2'] < 20.99) | (blood_gas_parsed['fio2'] > 100), 
                                    blood_gas_parsed['fio2'], np.nan)
blood_gas_parsed['peep'] = np.where((blood_gas_parsed['peep'] <=  0) | (blood_gas_parsed['peep'] >= 100), 
                                  blood_gas_parsed['peep'], np.nan)

blood_gas_parsed = blood_gas_parsed[usable_columns]
blood_gas_parsed.to_csv('mimiciii_blood_gas_cleaned.csv', index=False)

# eventually we'll need to interpolate so we should check out the average timedeltas for each patient 
timetable = blood_gas_parsed[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']]
timetable = timetable.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
timedelta = timetable.diff()['CHARTTIME']
timedelta.describe()

# finally get the dataframe with our interpolated data
interpolated_blood_gas = get_lastval_interpolation(blood_gas_parsed, interval = 10)