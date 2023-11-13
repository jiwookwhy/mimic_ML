-- Extract durations and dosages of anaesthetic medication, defined as:
-- Fentanyl: 30118, 30149 (Conc), 30150 (Base), 30308 (Drip), 221744, 225972 (Push), 225942 (Conc)
-- Propofol: 30131, 227210 (Intub), 222168
-- Midazolam: 30124, 221668
-- Dilaudid: 30163, 221833

with io_cv as
(
  select
    icustay_id, charttime, itemid, stopped, rate, rateuom, amount, amountuom
  from mimiciii.inputevents_cv
  where itemid in
  (
    30118, 30149, 30150, 30308 -- Fentanyl
    , 30131 -- Propofol
    , 30124 -- Midazolam
    , 30163 -- Dilaudid
  )
)
, drugcv1 as
(
  select
    icustay_id, charttime, itemid, rateuom, amountuom
    -- case statement determining whether the ITEMID is an instance of medication usage
    , 1 as drug

    -- the 'stopped' column indicates if a medication has been disconnected
    , max(case when stopped in ('Stopped','D/C''d') then 1
          else 0 end) as drug_stopped

    , max(case when rate is not null then 1 else 0 end) as drug_null
    , max(rate) as drug_rate
    , max(amount) as drug_amount

  from io_cv
  group by icustay_id, charttime, itemid, rateuom, amountuom
)
, drugcv2 as
(
  select v.*
    , sum(drug_null) over (partition by icustay_id, itemid order by charttime) as drug_partition
  from
    drugcv1 v
)
, drugcv3 as
(
  select v.*
    , first_value(drug_rate) over (partition by icustay_id, itemid, drug_partition order by charttime) as drug_prevrate_ifnull
  from
    drugcv2 v
)
, drugcv4 as
(
select
    icustay_id
    , charttime
    , itemid
    -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, drug order by charttime))) AS delta

    , drug
    , drug_rate
	, rateuom
    , drug_amount
	, amountuom
    , drug_stopped
    , drug_prevrate_ifnull

    -- We define start time here
    , case
        when drug = 0 then null

        -- if this is the first instance of the drug
        when drug_rate > 0 and
          LAG(drug_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, drug, drug_null
          order by charttime
          )
          is null
          then 1

        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes drugnum sequential
        when drug_rate = 0 and
          LAG(drug_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, drug
          order by charttime
          )
          = 0
          then 0

        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- drug_prevrate_ifnull is equal to the previous value *iff* the current value is null
        when drug_prevrate_ifnull = 0 and
          LAG(drug_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, drug
          order by charttime
          )
          = 0
          then 0

        -- If the last recorded rate was 0, newdrug = 1
        when LAG(drug_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, drug
          order by charttime
          ) = 0
          then 1

        -- If the last recorded drug was D/C'd, newdrug = 1
        when
          LAG(drug_stopped,1)
          OVER
          (
          partition by icustay_id, itemid, drug
          order by charttime
          )
          = 1 then 1

        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, drug order by charttime))) > (interval '4 hours') then 1
      else null
      end as drug_start

FROM
  drugcv3
)
-- propagate start/stop flags forward in time
, drugcv5 as
(
  select v.*
    , SUM(drug_start) OVER (partition by icustay_id, itemid, drug order by charttime) as drug_first
FROM
  drugcv4 v
)
, drugcv6 as
(
  select v.*
    -- We define end time here
    , case
        when drug = 0
          then null

        -- If the recorded drug was D/C'd, this is an end time
        when drug_stopped = 1
          then drug_first

        -- If the rate is zero, this is the end time
        when drug_rate = 0
          then drug_first

        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on medications
        -- in principle, this could add an extra end time for the medication
        -- however, since we later group on drug_start, any extra end times are ignored
        when LEAD(CHARTTIME,1)
          OVER
          (
          partition by icustay_id, itemid, drug
          order by charttime
          ) is null
          then drug_first

        else null
        end as drug_stop
    from drugcv5 v
)

--show results for carevue sys patients
select distinct cv6.icustay_id, cv6.charttime, cv6.itemid, cv6.drug, cv6.drug_rate, cv6.rateuom, cv6.drug_amount, cv6.amountuom
 , case when drug_stopped = 1 then 'Y' else '' end as stopped
 , drug_start
 , drug_first
 , drug_stop
 from drugcv6 as cv6
 inner join public.ventilation_classification as vc ON cv6.icustay_id = vc.icustay_id
 inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
 inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
 where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
 order by icustay_id, charttime;
