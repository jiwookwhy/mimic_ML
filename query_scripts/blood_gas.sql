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
