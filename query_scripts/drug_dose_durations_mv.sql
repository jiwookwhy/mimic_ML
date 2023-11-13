-- select only the ITEMIDs from the inputevents_mv table related to medications
with io_mv as
(
  select
    icustay_id, linkorderid, itemid, starttime, endtime, amount, amountuom, rate, rateuom
  from mimiciii.inputevents_mv io
  -- Subselect the medication ITEMIDs
  where itemid in
  (
    222168, --Propofol
    227210, --Propofol (Intubation)
    221668, --Midazolam (Versed)
    221833, --Hydromorphone (Dilaudid)
    221744, --Fentanyl
    225972, --Fentanyl (Push)
    225942  --Fentanyl (Concentrate)
  )
  and statusdescription != 'Rewritten' -- only valid orders
)

--show results for carevue sys patients
select distinct io.icustay_id, io.linkorderid, io.itemid, io.starttime, io.endtime, io.amount, io.amountuom, io.rate, io.rateuom
from io_mv as io
inner join public.ventilation_classification as vc ON io.icustay_id = vc.icustay_id
inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
order by icustay_id, starttime;
