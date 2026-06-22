# flow_finalv1.0

Production release copied from the validated `r2` model snapshot.

- PR target: observed weekly GPCP precipitation.
- T2M target: observed ERA5 T2M minus target-lead GEOS ensemble-mean T2M.
- GEOS inputs: mean, standard deviation, q10, q90, and member count.
- Includes static geography, global cross-attention, separate PR/T2M heads,
  structured gradient/multiscale losses, and the established velocity/variance schedule.
- Trains in 30-epoch scheduler sessions and resumes from `latest_flow_ckpt.pt`.

Submit training:

```bash
sbatch ml_model/submit_train_flow_finalv1_0.sh
```
