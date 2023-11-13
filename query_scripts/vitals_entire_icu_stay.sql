-- This query pivots the vital signs for entire icu stay of a patient
-- Vital signs include heart rate, blood pressure, respiration rate, and temperature

SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, pvt.valueuom

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
  select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime, ce.valueuom
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
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.vitalid, pvt.valuenum, pvt.charttime, pvt.valueuom
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime;
