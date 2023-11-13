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
from utils import mimic_parser, get_cohort_timings, get_cohort_filtered_data, get_time_filtered_data

#def some helpers for vitals cleaning
def clean_hr(df):
    """
    Cleans up heart rate
    """
    clean_hr = []
    for i in df['heartrate']:
        #make any unphysiological vals nan
        #am being very generours here - allowing for coverage of both of a marathon runner and
        #someone very sick and possibly about to die
        if i<38 or i>300:
            i = np.nan
        clean_hr.append(i)
    df['heartrate'] = pd.Series(clean_hr)
    return df

def clean_sysbp(df):
    """
    Cleans up systolic BP
    """
    clean_sysbp = []
    for i in df['sysbp']:
        #am cutting here only very small (unphysiological) values
        #like a sysbp below 20...
        #but am leaving upper range (have seen some really scary high vals in real life here)
        #and patient was still alive
        if i<20:
            i = np.nan
        clean_sysbp.append(i)
    df['sysbp'] = pd.Series(clean_sysbp)
    return df

def clean_diasbp(df):
    """
    Cleans up diastolic BP
    """
    clean_diasbp = []
    for i in df['diasbp']:
        #am cutting here only very small (unphysiological) values
        #like a diasbp below 20...
        #but am leaving upper range as before
        if i<20:
            i = np.nan
        clean_diasbp.append(i)
    df['diasbp'] = pd.Series(clean_diasbp)
    return df

def clean_glucose(df):
    """
    Cleans up glucose measurements
    """
    clean_gluc = []
    for i in df['glucose']:
        #No units were logged for these!
        #very low vals would make more sense if in mmol/L (vs. mg/dL)
        #but we can't check if this is the case.
        #So am mostly imposing lower range (based on what I know it's barely survivable)
        #2 is already super low (normal range ~90-110 mg/dL // 3-7(ish)mmol/L in adults)
        #and am not imposing upper range b/c I've seen some VERY HIGH SCARY vals here
        #for DM patients.
        if i<2.00:
            i = np.nan
        clean_gluc.append(i)
    df['glucose'] = pd.Series(clean_gluc)
    return df


def clean_spo2(df):
    """
    Cleans up pulse ox measurements
    """
    clean_spo2 = []
    for i in df['SpO2']:
        #pulse ox measurements tend to be pretty unreliable
        #especially for finger devices
        #so will only take things that make sense
        #i.e., between 10-100. Everything else is prolly garbage measures
        if i<10.00 or i>100:
            i = np.nan
        clean_spo2.append(i)
    df['SpO2'] = pd.Series(clean_spo2)
    return df

def clean_temp(df):
    """
    Cleans up temp measurements.
    These are already in Celsius.
    """
    #cutoffs based on what level or hypo or hyperthermia
    #that is generally considered lethal
    clean_temp = []
    for i in df['temp']:
        if i<22 or i>45:
            i  = np.nan
        clean_temp.append(i)
    df['temp'] = pd.Series(clean_temp)
    return df

def clean_resprate(df):
    """
    Cleans up respiratory rate measurements.
    """
    #Here am just cutting anything below 1
    #These are typically measured on a /min basis so 1 would be very low
    #but ok if you had some especial circumstances
    #higher ones are also believable (pat stressed, pat acidic or just high vent setting)
    clean_rr = []
    for i in df['resprate']:
        if i<1:
            i  = np.nan
        clean_rr.append(i)
    df['resprate'] = pd.Series(clean_rr)
    return df


#now let's actually clean up data
#read in vitals raw query file
vitals_raw = pd.read_csv('../../raw_query_data/mimiciii_vitals.csv')
#parse raw file
keys = ['subject_id', 'hadm_id', 'icustay_id', 'charttime']
vitals_parsed = mimic_parser(keys, vitals_raw)
#get only cols we want
vitals_parsed = vitals_parsed[['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'heartrate', 'sysbp', 'diasbp', \
'resprate', 'temp', 'SpO2', 'glucose']]

#clean up cols we got
vitals_clean = clean_hr(vitals_parsed)
vitals_clean = clean_sysbp(vitals_clean)
vitals_clean = clean_diasbp(vitals_clean)
vitals_clean = clean_glucose(vitals_clean)
vitals_clean = clean_spo2(vitals_clean)
vitals_clean = clean_temp(vitals_clean)
vitals_clean = clean_resprate(vitals_clean)

#now filter by cohort and time
#load files
cohort_info = pd.read_csv('../../raw_query_data/cohort_info_CORRECTED.csv')
vent_durations_info = pd.read_csv('../../raw_query_data/vent_duration_info.csv')
icu_timings_info = pd.read_csv('../../raw_query_data/icustay_InOutTime_info.csv')

#get cohort and timings to filter with
cohort = pd.unique(cohort_info['icustay_id'])
cohort_times = get_cohort_timings(cohort, vent_durations_info, icu_timings_info)

#filter
filtered_clean_vitals = get_cohort_filtered_data(vitals_clean, cohort)
filtered_clean_vitals = get_time_filtered_data(filtered_clean_vitals, cohort_times)

#and save this out!
out_dir= '../../clean_forGP_data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
filtered_clean_vitals.to_csv(os.path.join(out_dir, 'vitals.csv'))
