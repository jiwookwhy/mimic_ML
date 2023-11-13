# Does some basic clean up for vent settings data
# Mostly does:
# 1) Makes fIO2 and peep values are within reasonable (physiological/possible) ranges
# 2) Picks only cols we want (subj_id, hadm_od, icustay_id, charttimes, fio2, peep, rass)
# 3) Filters data to include ONLY icu_ids in cohort file (CORRECTED file)
# 4) Filters data to include ONLY charttimes between start of first mech vent event and end of icu icustay
# Results from this file will be saved and used for actual GP fitting and final feature extraction

import numpy as np
import pandas as pd
import os
from utils import mimic_parser, get_cohort_timings, get_time_filtered_data

#def some helpers
def clean_fio2(df):
    clean_fio2 = []
    for i in df['fio2']:
        if i>0 and i<=1:
            i = i*100 #convert to %
        elif i<20.99 or i>100:
            i = np.nan #throw away non-phisiological values
        clean_fio2.append(i)
    df['fio2'] = pd.Series(clean_fio2)
    return df

def clean_peep(df):
    clean_peep = []
    for i in df['peep']:
        if i<=0:
            i = np.nan
        elif i>=100:
            i = np.nan
        clean_peep.append(i)
    df['peep'] = pd.Series(clean_peep)
    return df


#now let's actually clean up data
#read in vent settings raw query file
vent_raw = pd.read_csv('../../raw_query_data/mimiciii_ventilator_sedation.csv')
#parse raw file
keys = ['subject_id', 'hadm_id', 'icustay_id', 'charttime']
vent_parsed = mimic_parser(keys, vent_raw)
#get only cols we want
vent_parsed = vent_parsed[['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'fio2', 'peep', 'rass']]
#clean up fiO2 and peep cols
#rass scores are actually ok (i.e., within correct ranges!)
vent_clean = clean_fio2(vent_parsed)
vent_clean = clean_peep(vent_clean)
#now get data time filtered
#note that, for this file icustay_ids are EXACTLY the ones in our cohort
#so am skipping step filtering by cohort icustay_id here!
cohort = pd.unique(vent_clean['icustay_id'])
vent_durations_info = pd.read_csv('../../raw_query_data/vent_duration_info.csv')
icu_timings_info = pd.read_csv('../../raw_query_data/icustay_InOutTime_info.csv')
cohort_times = get_cohort_timings(cohort, vent_durations_info, icu_timings_info)
filtered_clean_vent = get_time_filtered_data(vent_clean, cohort_times)
#finally save this
out_dir= '../../clean_forGP_data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
filtered_clean_vent.to_csv(os.path.join(out_dir, 'vent_settings.csv'))
