--am taking these straight from concepts table named ventilation_classification
--in there, they already have cols for extubation, selfextubation and niO2therapy events!
select vc.icustay_id, vc.charttime, vc.extubated, vc.selfextubated, vc.oxygentherapy
from public.ventilation_classification as vc
inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by vc.icustay_id, vc.charttime, vc.extubated, vc.selfextubated, vc.oxygentherapy
order by vc.icustay_id, vc.charttime
