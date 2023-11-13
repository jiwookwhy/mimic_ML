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
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime;
