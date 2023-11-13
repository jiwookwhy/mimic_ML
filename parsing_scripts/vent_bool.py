import pandas as pd
import numpy as np

# vent_label adds a column of booleans (onvent) to df that is 1 if on ventilation 
# and 0 otherwise

# args 
# df [dataframe]: dataframe to add the onvent boolean column to
# reference_df [dataframe]: dataframe containing icustayid as well as
# columns containing start and end times for ventilation 
# start_col [string]: name of column containing the start of ventilation
# end_col [string]: name of column containing the end of ventilation
    
def vent_label(df, reference_df, start_col, end_col):
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
    
    reference_df = reference_df[['icustay_id', start_col, end_col]]
    reference_df[start_col] = pd.to_datetime(reference_df[start_col])
    reference_df[end_col] = pd.to_datetime(reference_df[end_col])
    
    # use groupby on icustay_id with np.which combined with within group ordering
    # to perform a logical intersection via taking max on boolean 
    # for ['CHARTTIME', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'] instance
    # to get the correct boolean column
    reference_df = reference_df.merge(df[['CHARTTIME', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']], right_on = 'ICUSTAY_ID', left_on = 'icustay_id')
    mask = (reference_df[start_col] <= reference_df['CHARTTIME']) & (reference_df['CHARTTIME'] <= reference_df[end_col])
    reference_df['onvent'] = np.where(mask, 1, 0)
    reference_df = reference_df.drop(axis = 1, labels =['icustay_id', start_col, end_col])
    reference_df = reference_df.groupby(['CHARTTIME', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    # use max/lattice join trick with boolean column
    reference_df = reference_df.max().reset_index()
    
    # merge with reference_df that now contains the correct onvent column
    df = df.merge(reference_df, how = 'left',  on = ['CHARTTIME', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    return(df)
    
    


df = pd.read_csv('mimiciii_blood_gas_cleaned.csv')

reference_df = pd.read_csv('vent_duration_info.csv', index_col = 0)
start_col = 'starttime'
end_col = 'endtime'

result = vent_label(df, reference_df, start_col, end_col)