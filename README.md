# learning_to_re_adapt


```bash
python -m scripts.render_experiment outputs/2025-10-23/ppo_hopper_sanity_1

python -m scripts.run_experiment configs/ppo.yaml  

```

on cluster 

```bash
sbatch run.sbatch configs/ppo_hopper.yaml
```