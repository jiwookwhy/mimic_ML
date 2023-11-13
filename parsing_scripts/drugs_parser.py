import os
import datetime

import pandas as pd
import numpy as np

from utils import get_cohort_timings
from config import Config
from tqdm import tqdm

config = Config()
data_dir = config.data_path


def get_cohort(cohort_info_path):
    """
    Returns all unique icustay_id's as a list
    """
    df_cohort = pd.read_csv(cohort_info_path)
    return df_cohort["icustay_id"].unique()


cv_codes = {
    30118: "fentanyl",
    30149: "fentanyl", # DRIP
    30150: "fentanyl", # BOLUS IN mL
    30308: "fentanyl",
    30131: "propofol",
    30124: "midazolam",
    30163: "hydromorphone",
}

mv_codes = {
    222168: "propofol",
    # 227210: "propofol (intubation)", # NOT PRESENT IN THE DATA
    221668: "midazolam",
    221833: "hydromorphone", # ONLY GIVEN AS BOLUS
    221744: "fentanyl", # ASSOCIATE WITH BOLUS INJECTIONS
    225972: "fentanyl (push)",
    225942: "fentanyl (concentrate)",
}

# File paths
print("Loading CSVs")
vent_info_path = os.path.join(data_dir, "vent_duration_info.csv")
cohort_info_path = os.path.join(data_dir, "cohort_info.csv")
icu_timings_info_path = os.path.join(data_dir, "icustay_InOutTime_info.csv")
mv_info_path = os.path.join(data_dir, "mimiciii_drugs_mv.csv")
cv_info_path = os.path.join(data_dir, "mimiciii_drugs_cv.csv")

# Load tables
df_vent = pd.read_csv(vent_info_path)
df_timings = pd.read_csv(icu_timings_info_path)
df_mv = pd.read_csv(mv_info_path)
df_cv = pd.read_csv(cv_info_path)
print("CSVs loaded")

# Load cohort, ids
cohort = get_cohort(cohort_info_path)
mv_ids = df_mv["icustay_id"].unique()
print(len(mv_ids))
cv_ids = df_cv["icustay_id"].unique()
print(len(cv_ids))

# Get timings for filtering
cohort_times = get_cohort_timings(cohort, df_vent, df_timings)

df_total = pd.DataFrame(
    columns=["icustay_id", "charttime"]
)  # drug columns will be added dynamically based on if they exist in the data


# concatenate dataframes and reset index
def concat_reset(*args):
    df = pd.concat([*args], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df


# add a row to CareVue dataframe based on the name of the drug corresponding to itemid
def add_cv_row(df, entry):
    drug_name = cv_codes[entry.itemid]
    new_row = {
        "icustay_id": [entry.icustay_id],
        "charttime": [entry.charttime],
    }
    new_row[drug_name] = [entry.drug_amount]
    return concat_reset(df, pd.DataFrame(new_row))


# add row of with empty drug info for specific icustay_id and time (a datetime object)
def add_empty_row(df, icustay_id, time):
    new_row = {
        "icustay_id": [icustay_id],
        "charttime": [time],
    }
    return concat_reset(df, pd.DataFrame(new_row))


for icustay_id in tqdm({i: cohort_times[i] for i in cohort_times if i < 200_100}):
    # Match icustay_id to either mv or cv drug dataframe
    start_time, end_time = cohort_times[icustay_id]
    in_mv = icustay_id in mv_ids
    in_cv = icustay_id in cv_ids

    # cv is easier:
    if in_cv:
        # THIS NEED TO BE UPDATED WITH NEW SQL CODE

        # filter based on specific icustay
        df_temp = df_cv[df_cv["icustay_id"] == icustay_id]

        # (1) check if time is on the hour -- if not, drop it (all "off-hour"
        # times are duplicates or just NaN rows)
        # convert times to pandas datetime objs
        df_temp["charttime"] = pd.to_datetime(df_temp["charttime"])
        df_temp = df_temp[df_temp["charttime"].dt.minute == 0]

        # (2) if drug amount is zero or empty, record amount as 0
        # otherwise, record the amount as the drug amount
        # - NaN are interpreted as zero which is the correct medical interpretation (I think)
        # - This could be slow...
        for index, entry in df_temp.iterrows():
            df_total = add_cv_row(df_total, entry)

    # mv could be harder:
    elif in_mv:  # metavision
        pass
        # (1) remove all bolus events
        # (1) for each time on the hour from first ventilation to exiting icu
        #     take closest sample from timespan on drug
    else:  # in neither dataset, just fill all times between the cohort_times[icustay_id]
        start, end = cohort_times[icustay_id]
        for time in pd.period_range(start=start, end=end, freq="H"):
            df_total = add_empty_row(df_total, icustay_id, time)


df_total = df_total.replace(np.nan, 0.0)
print(df_total)
df_total.to_csv("out.csv", index=False)
