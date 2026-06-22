# flow_finalv1_global

Full-global counterpart of the validated `flow_finalv1.0` South Asia release.

- Target grid: 181x360 at 1 degree, from 90S-90N and 0E-359E.
- PR target: observed weekly GPCP precipitation.
- T2M target: observed ERA5 T2M minus target-lead GEOS ensemble-mean T2M.
- GEOS inputs: mean, standard deviation, q10, q90, and member count.
- Predictor variables, architecture, losses, optimization, and velocity/variance
  schedules are unchanged from `flow_finalv1.0`.
- Static geography is rebuilt on the global grid. GLDAS-uncovered polar rows are
  assigned physical sea-level elevation rather than extrapolated terrain.
- Training starts from scratch, runs 10 epochs per scheduler session, and resumes
  from `latest_flow_ckpt.pt`.

Submit training:

```bash
cd /scratch/11353/afahad/geossub/geos_subc
git pull
mkdir -p ml_output_flow_finalv1_global_noisectx_t2mres
sbatch ml_model/submit_train_flow_finalv1_global.sh
```
