import psycopg2
from psycopg2 import OperationalError
import pandas as pd

db_name='mimic'
db_user = 'daniela'
db_password = 'olivera'
db_host = 'localhost'
db_port = '5432'

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

#test creating connection
conn = create_connection(db_name, db_user, db_password, db_host, db_port)

#constructs queries we want

#cohort
#entries w/same ICU stay are d/t re-intubations here
cohort_query = """
SELECT DISTINCT de.subject_id, de.hadm_id, vc.icustay_id, de.hospital_expire_flag, vc.mechvent,
de.first_icu_stay, vd.duration_hours, vd.starttime, vd.endtime
FROM public.ventilation_classification AS vc INNER JOIN public.ventilation_durations as vd ON vc.icustay_id=vd.icustay_id
INNER JOIN public.icustay_detail as de ON de.icustay_id=vd.icustay_id
WHERE vc.mechvent=1 AND de.first_icu_stay='true' AND vd.duration_hours > 24 AND de.hospital_expire_flag <> 1
order by vc.icustay_id, vd.starttime
"""

#demographics
#seems like we are missing weight info for some of these
demo_query = """
SELECT DISTINCT de.subject_id, de.hadm_id, de.icustay_id,
de.gender, de.admission_age, de.ethnicity, wfd.weight_admit
FROM public.icustay_detail as de INNER JOIN public.ventilation_classification AS vc ON de.icustay_id=vc.icustay_id
INNER JOIN public.ventilation_durations as vd ON vd.icustay_id=vc.icustay_id
INNER JOIN public.weight_first_day as wfd ON wfd.icustay_id = vd.icustay_id
WHERE vc.mechvent=1 AND de.first_icu_stay='true' AND vd.duration_hours > 24 AND de.hospital_expire_flag <> 1
order by de.icustay_id
"""

#adapted from concepts script for
#agregate vitals vals over 24hr period
#here we DO NOT aggregate and report all entries over entire icustay
#this is being run on BigQuery
#only diff is that I added filtering
vitals_query = """
-- This query pivots the vital signs for entire icu stay of a patient
-- Vital signs include heart rate, blood pressure, respiration rate, and temperature

SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime

-- Easier names
, case when VitalID = 1 then valuenum ELSE NULL END AS heartrate
, case when VitalID = 2 then valuenum ELSE NULL END AS sysbp
, case when VitalID = 3 then valuenum ELSE NULL END AS diasbp
, case when VitalID = 4 then valuenum ELSE NULL END AS meanbp
, case when VitalID = 5 then valuenum ELSE NULL END AS resprate
, case when VitalID = 6 then valuenum ELSE NULL END AS temp
, case when VitalID = 7 then valuenum ELSE NULL END AS SpO2
, case when VitalID = 8 then valuenum ELSE NULL END AS glucose

FROM  (
  select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime
  , case
    when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 1 -- HeartRate
    when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 2 -- SysBP
    when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 3 -- DiasBP
    when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 -- MeanBP
    when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 5 -- RespRate
    when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 6 -- TempF, converted to degC in valuenum call
    when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 6 -- TempC
    when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 7 -- SpO2
    when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 8 -- Glucose

    else null end as vitalid
      -- convert F to C
  , case when itemid in (223761,678) then (valuenum-32)/1.8 else valuenum end as valuenum

  from mimiciii.icustays as ie
  inner join public.ventilation_classification as vc ON vc.icustay_id = ie.icustay_id
  inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
  inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
  left join mimiciii.chartevents ce
  on de.icustay_id = ce.icustay_id
  and ce.charttime between ie.intime and ie.outtime --get all entries for entire icu_stay
  -- exclude rows marked as error
  and (ce.error IS NULL or ce.error = 0)
  where ce.itemid in
  (
  -- HEART RATE
  211, --"Heart Rate"
  220045, --"Heart Rate"

  -- Systolic/diastolic

  51, --	Arterial BP [Systolic]
  442, --	Manual BP [Systolic]
  455, --	NBP [Systolic]
  6701, --	Arterial BP #2 [Systolic]
  220179, --	Non Invasive Blood Pressure systolic
  220050, --	Arterial Blood Pressure systolic

  8368, --	Arterial BP [Diastolic]
  8440, --	Manual BP [Diastolic]
  8441, --	NBP [Diastolic]
  8555, --	Arterial BP #2 [Diastolic]
  220180, --	Non Invasive Blood Pressure diastolic
  220051, --	Arterial Blood Pressure diastolic


  -- MEAN ARTERIAL PRESSURE
  456, --"NBP Mean"
  52, --"Arterial BP Mean"
  6702, --	Arterial BP Mean #2
  443, --	Manual BP Mean(calc)
  220052, --"Arterial Blood Pressure mean"
  220181, --"Non Invasive Blood Pressure mean"
  225312, --"ART BP mean"

  -- RESPIRATORY RATE
  618,--	Respiratory Rate
  615,--	Resp Rate (Total)
  220210,--	Respiratory Rate
  224690, --	Respiratory Rate (Total)


  -- SPO2, peripheral
  646, 220277,

  -- GLUCOSE, both lab and fingerstick
  807,--	Fingerstick Glucose
  811,--	Glucose (70-105)
  1529,--	Glucose
  3745,--	BloodGlucose
  3744,--	Blood Glucose
  225664,--	Glucose finger stick
  220621,--	Glucose (serum)
  226537,--	Glucose (whole blood)

  -- TEMPERATURE
  223762, -- "Temperature Celsius"
  676,	-- "Temperature C"
  223761, -- "Temperature Fahrenheit"
  678 --	"Temperature F"

  )
  and vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
) as pvt
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.vitalid, pvt.valuenum, pvt.charttime
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
"""

#add query for ventilation parameters
#this was run on BiGQuery
vent_query = """
SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime

-- Easier names
, case when VentID = 1 then valuenum ELSE NULL END AS peep
, case when VentID = 2 then valuenum ELSE NULL END AS fio2
, case when VentID = 3 then valuenum ELSE NULL END AS minutevol
, case when VentID = 4 then valuenum ELSE NULL END AS tidalvol
, case when VentID = 5 then valuenum ELSE NULL END AS resppress
, case when VentID = 6 then valuenum ELSE NULL END AS insppress
, case when VentID = 7 then valuenum ELSE NULL END AS rass

FROM  (
  select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime
  , case
    when itemid in (60,437,505,506,686,220339,224700) then 1 -- peep
    when itemid in (3420, 190, 223835, 3422, 727) then 2 -- fio2
    when itemid in (445, 448, 449, 450, 1340, 1486, 1600, 224687) then 3 -- minutevol
    when itemid in (639, 654, 681, 682, 683, 684,224685,224684,224686) then 4 -- tidalvol
    when itemid in (218,436,535,444,224697,224695,224696,224746,224747) then 5 -- resppress
    when itemid in (221,1,1211,1655,2000,226873,224738,224419,224750,227187) then 6 -- insppress
    when itemid in (228096) then 7 -- rass

    else null end as VentID
  , case when itemid in (228096) then valuenum else valuenum end as valuenum
  from mimiciii.icustays as ie
  inner join public.ventilation_classification as vc ON vc.icustay_id = ie.icustay_id
  inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
  inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
  left join mimiciii.chartevents ce
  on de.icustay_id = ce.icustay_id
  and ce.charttime between ie.intime and ie.outtime --get all entries for entire icu_stay
  -- exclude rows marked as error
  and (ce.error IS NULL or ce.error = 0)
  where ce.itemid in
  (
  --peep
  60,
  437,
  505,
  506,
  686,
  220339,
  224700,

  -- fio2
  3420,
  190,
  223835,
  3422,
  727,

  -- minutevol
  445,
  448,
  449,
  450,
  1340,
  1486,
  1600,
  224687,

  --tidalvol
  639,
  654,
  681,
  682,
  683,
  684,
  224685,
  224684,
  224686,

  --resppress
  218,
  436,
  535,
  444,
  224697,
  224695,
  224696,
  224746,
  224747,

  --insppress
  221,
  1,
  1211,
  1655,
  2000,
  226873,
  224738,
  224419,
  224750,
  227187,

  --rass
  228096

  )
  and vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
) as pvt
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.ventid, pvt.valuenum, pvt.charttime
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime"""

##query for blood gas measurements
bg_query = """
-- The aim of this query is to pivot entries related to blood gases and
-- chemistry values which were found in LABEVENTS

-- things to check:
--  when a mixed venous/arterial blood sample are taken at the same time, is the store time different?

with pvt as
( -- begin query that extracts the data
  select ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
  --codes here are diff b/c they refer to labs table... Am picking these as well so we can have
  -- all possible measurements of a given quantity
      , case
        when itemid = 50800 then 'SPECIMEN'
        when itemid = 50801 then 'AADO2'
        when itemid = 50802 then 'BASEEXCESS'
        when itemid = 50803 then 'BICARBONATE'
        when itemid = 50804 then 'TOTALCO2'
        when itemid = 50812 then 'INTUBATED'
        when itemid = 50813 then 'LACTATE'
        when itemid = 50814 then 'METHEMOGLOBIN'
        when itemid = 50815 then 'O2FLOW'
        when itemid = 50816 then 'FIO2'
        when itemid = 50817 then 'SO2' -- OXYGENSATURATION
        when itemid = 50818 then 'PCO2'
        when itemid = 50819 then 'PEEP'
        when itemid = 50820 then 'PH'
        when itemid = 50821 then 'PO2'
        when itemid = 50823 then 'REQUIREDO2'
        when itemid = 50826 then 'TIDALVOLUME'
        when itemid = 50827 then 'VENTILATIONRATE'
        when itemid = 50828 then 'VENTILATOR'
        else null
        end as label
        , value
        -- add in some sanity checks on the values
        , case
          when valuenum <= 0 and itemid != 50802 then null -- allow negative baseexcess
          -- ensure FiO2 is a valid number between 21-100
          -- mistakes are rare (<100 obs out of ~100,000). GOOOD TO KNOW -- screen ours for it too!
          -- there are 862 obs of valuenum == 20 - some people round down!
          -- rather than risk imputing garbage data for FiO2, we simply NULL invalid values
          when itemid = 50816 and valuenum < 20 then null
          when itemid = 50816 and valuenum > 100 then null
          when itemid = 50817 and valuenum > 100 then null -- O2 sat
          when itemid = 50815 and valuenum >  70 then null -- O2 flow
          when itemid = 50821 and valuenum > 800 then null -- PO2
           -- conservative upper limit
        else valuenum
        end as valuenum
    FROM mimiciii.icustays as ie
    inner join public.ventilation_classification as vc ON vc.icustay_id = ie.icustay_id
    inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
    inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
    left join mimiciii.labevents as le
      on le.subject_id = ie.subject_id and le.hadm_id = ie.hadm_id
      and le.charttime between ie.intime and ie.outtime
      and le.ITEMID in
      -- blood gases
      (
        50800, 50801, 50802, 50803, 50804, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819
        , 50820, 50821, 50822, 50823, 50826, 50827, 50828
      )
      and vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
)
select pvt.SUBJECT_ID, pvt.HADM_ID, pvt.ICUSTAY_ID, pvt.CHARTTIME
, case when label = 'SPECIMEN' then value else null end as specimen
, case when label = 'AADO2' then valuenum else null end as aado2
, case when label = 'BASEEXCESS' then valuenum else null end as baseexcess
, case when label = 'BICARBONATE' then valuenum else null end as bicarbonate
, case when label = 'TOTALCO2' then valuenum else null end as totalco2
, case when label = 'INTUBATED' then valuenum else null end as intubated
, case when label = 'LACTATE' then valuenum else null end as lactate
, case when label = 'METHEMOGLOBIN' then valuenum else null end as methemoglobin
, case when label = 'O2FLOW' then valuenum else null end as o2flow
, case when label = 'FIO2' then valuenum else null end as fio2
, case when label = 'SO2' then valuenum else null end as so2 -- OXYGENSATURATION
, case when label = 'PCO2' then valuenum else null end as pco2
, case when label = 'PEEP' then valuenum else null end as peep
, case when label = 'PH' then valuenum else null end as ph
, case when label = 'PO2' then valuenum else null end as po2
, case when label = 'REQUIREDO2' then valuenum else null end as requiredo2
, case when label = 'TIDALVOLUME' then valuenum else null end as tidalvolume
, case when label = 'VENTILATIONRATE' then valuenum else null end as ventilationrate
, case when label = 'VENTILATOR' then valuenum else null end as ventilator
from pvt
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME, pvt.label, pvt.value, pvt.valuenum
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME;
"""

extubation_flags_query = """
select vc.icustay_id, vc.charttime, vc.extubated, vc.selfextubated, vc.oxygentherapy
from public.ventilation_classification as vc
inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by vc.icustay_id, vc.charttime, vc.extubated, vc.selfextubated, vc.oxygentherapy
order by vc.icustay_id, vc.charttime
"""

vent_durations_query = """
select vd.icustay_id, vd.ventnum, vd.starttime, vd.endtime, vd.duration_hours
from public.ventilation_durations as vd
inner join public.ventilation_classification as vc ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by vd.icustay_id, vd.ventnum, vd.starttime, vd.endtime, vd.duration_hours
order by vd.icustay_id, vd.starttime
"""

icu_in_out_times_query = """
select ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime
from mimiciii.icustays as ie
inner join public.ventilation_classification as vc ON vc.icustay_id = ie.icustay_id
inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime
order by ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime
"""

#get cohort df and save it
res = execute_read_query(conn, cohort_query)
df = pd.DataFrame(res, columns=['subj_id', 'hadm_id', 'icustay_id', 'hx_exp', 'mechvent', 'first_icu_stay', \
'duration_hrs', 'starttime', 'endtime',])
#report numer of unique patients in here (this should match more of less what's reported on paper)
print('Number of unique patients in our cohort: {}'.format(len(pd.unique(df['subj_id']))))
print('Number of unique ICU stays in cohort:{}'.format(len(pd.unique(df['icustay_id']))))
df.to_csv('./cohort_info.csv')

#get demographics info and save it
res = execute_read_query(conn, demo_query)
df = pd.DataFrame(res, columns=['subj_id', 'hadm_id', 'icustay_id', 'gender', \
'admission_age', 'ethnicity', 'weight_admit'])
df.to_csv('./demographics_info.csv')
print('Finished and saved demographics query!')

#get extubation flag info
res = execute_read_query(conn, extubation_flags_query)
df = pd.DataFrame(res, columns=['icustay_id', 'charttime', \
'extubated', 'selfextubated', 'oxygentherapy'])
df.to_csv('./extubation_flag_info.csv')
print('Finished and saved extubation_flags query!')

#get extubation duration info
res = execute_read_query(conn, vent_durations_query)
df = pd.DataFrame(res, columns=['icustay_id', 'ventnum', \
'starttime', 'endtime', 'duration_hours'])
df.to_csv('./vent_duration_info.csv')
print('Finished and saved vent_durations query!')

#get info on when patient gets into and out of icu
#get extubation duration info
res = execute_read_query(conn, icu_in_out_times_query)
df = pd.DataFrame(res, columns=['subj_id', 'hadm_id', 'icustay_id', 'intime', 'outtime'])
df.to_csv('./icustay_InOutTime_info.csv')
print('Finished and saved ICU In/Out time info query!')
