import pandas as pd

# keys [list]: any set of candidate keys 
# df [dataframe]: mimic data as pandas dataframe
def mimic_parser(keys, df):
    # get all the non-key columns
    non_key_cols = list(set(df.columns) - set(keys))
    # table of just keys
    parsed_df = df[keys].drop_duplicates(subset = keys)
    
    for col in non_key_cols:
        # parse non-key cols one at a time and left join them
        temp = df[keys + [col]].dropna(axis = 0, subset = keys + [col])
        parsed_df = parsed_df.merge(temp, on = keys, how = 'left')
    
    parsed_df = parsed_df.loc[pd.notna(parsed_df.CHARTTIME)]
    return(parsed_df)

def get_cohort_timings(cohort, vent_durations, icu_timings):
    cohort_times={}
    for i in cohort:
        #pick starttime for very first vent event
        starttime = list(vent_durations.loc[vent_durations['icustay_id']==i, 'starttime'])[0]
        #pick end icu time
        endtime = list(icu_timings.loc[icu_timings['icustay_id']==i, 'outtime'])[0]
        item_timings = (starttime, endtime)
        cohort_times[i] = item_timings
    return cohort_times

def get_time_filtered_data(df, cohort_times):
    filtered_df = []
    for row in df.itertuples():
        id_times = cohort_times[row.icustay_id]
        if row.charttime >= id_times[0] and row.charttime<=id_times[1]:
            filtered_df.append(row)
        else:
            pass
    return pd.DataFrame(filtered_df)

def get_cohort_filtered_data(df, cohort_ids):
    filtered_df = []
    for row in df.itertuples():
        if row.icustay_id in cohort_ids:
            filtered_df.append(row)
        else:
            pass
    return pd.DataFrame(filtered_df)

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
    
    

# RESAMPLING METHODS
# -----------------------------------------------------------------------------
def resample(df, timestamp_col_name, columns: list[str], rate: int, method: str="pad"):
    """ Resample using "method"--fill new intermediate values either using a hold
    ("pad") or linear interpolation ("interpolate")

    Args:
        df: dataframe of icu features at 1-hour intervals
        timestamp_col_name: name of column with timestamps
        columns: columns to resample
        rate: resampling rate in minutes (int)
        method: "pad" for hold, or "interpolate" for interpolate

    Returns:
        A sub-dataframe of the requested resampled columns
    """
    # check minutes divide an hour evenly (this seems to make life easier)
    assert method in ["pad", "interpolate"], "invalid requested resampling method"
    assert rate < 60 and rate > 0 and 60 % rate == 0, "invalid resample rate (minutes)"

    # filter to columns of interest, set indices to times
    df_filtered = df[[timestamp_col_name] + columns]
    df_timed = df_filtered.set_index(df_filtered[timestamp_col_name])

    # apply hold resampling at `rate` minutes, reset timestamp_col_name column
    if method == "pad":
        df_resampled = df_timed.resample(f"{rate}T").pad()
    else: # interpolate
        df_resampled = df_timed.resample(f"{rate}T").asfreq()
        df_resampled[columns] = df_resampled[columns].interpolate()
    df_resampled[timestamp_col_name] = df_resampled.index
    df_resampled = df_resampled.reset_index(drop=True)

    return df_resampled

# create dataframe with ten-minute time intervals (default) based on using 
# the patients first and last time in the dataframe with the last-value based 
# interpolated data 
# Idea is to do it in three steps
# (1) create a shell dataframe of ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']
# where the difference between charttimes is ten minutes (or any other desired frequency)
# for each ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'] group
# (2) full join the cleaned mimic data onto this shell dataframe 
# (3) use pandas to last-value interpolate features to get our dataframe with the desired frequency 
#
# args: 
# df [dataframe]: cleaned dataframe where ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']
# serve as candidate keys.
# interpolants_only [boolean]: set to true if you want only points where the data is interpolated 
# interval [int]: number of minutes to seperate interpolated data points
def get_lastval_interpolation(df, interval = 10):
    # get rid of all Null dates and also IDs
    df = df.dropna(axis = 0, subset = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME'])
    
    # make sure CHARTTIME col contains a timestamp object
    blood_gas_parsed['CHARTTIME'] = pd.to_datetime(blood_gas_parsed.CHARTTIME)
    
    keys = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']
    
    # sort df by keys 
    df = df.sort_values(keys)
    
    # first extract for each ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'] last time seen in the database
    time_ranges = df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME']].groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    time_ranges = time_ranges.agg({'CHARTTIME': ['min', 'max']}).reset_index()
    time_ranges.columns = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'min', 'max']
    
    # create list of dataframes for each patient with the desired time interval rows
    freq_str = str(interval)+'min'
    df_list = []
    for i in range(0, time_ranges.shape[0]):
        # get start and end times for patient in the dataset
        start = time_ranges.at[i,'min'].round(freq_str) 
        end = time_ranges.at[i,'max'].round(freq_str) 
        
        # create a dataframe for that patient and fill with chartimes with desired time interval between rows
        df_list.append(pd.DataFrame({'SUBJECT_ID': time_ranges.SUBJECT_ID[i],
                'HADM_ID': time_ranges.HADM_ID[i],
                'ICUSTAY_ID': time_ranges.ICUSTAY_ID[i],
                'CHARTTIME': pd.date_range(start = start, end = end, freq = freq_str)}))
        
    template_df = pd.concat(df_list)
    
    # do full outer join with data from mimic

    interpolated_df = template_df.merge(df, how = 'outer')
    interpolated_df = interpolated_df.sort_values(keys)
    interpolated_df = interpolated_df.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).ffill().reset_index()
     
    return interpolated_df
