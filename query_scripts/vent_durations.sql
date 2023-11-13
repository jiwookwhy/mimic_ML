select vd.icustay_id, vd.ventnum, vd.starttime, vd.endtime, vd.duration_hours
from public.ventilation_durations as vd
inner join public.ventilation_classification as vc ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by vd.icustay_id, vd.ventnum, vd.starttime, vd.endtime, vd.duration_hours
order by vd.icustay_id, vd.starttime
