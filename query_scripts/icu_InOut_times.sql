select ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime
from mimiciii.icustays as ie
inner join public.ventilation_classification as vc ON vc.icustay_id = ie.icustay_id
inner join public.ventilation_durations as vd ON vd.icustay_id = vc.icustay_id
inner join public.icustay_detail as de ON de.icustay_id = vd.icustay_id
where vc.mechvent=1 and vd.duration_hours > 24 and de.hospital_expire_flag <> 1 and de.first_icu_stay='true'
group by ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime
order by ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime
