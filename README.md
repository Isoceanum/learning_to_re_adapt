# learning_to_re_adapt


```bash

python -m scripts.run_experiment configs/mb_mpc.yaml  

python -m scripts.run_experiment configs/grbal.yaml  

python -m scripts.render_experiment outputs/2025-10-23/ppo_hopper_sanity_1

python -m scripts.run_tests

python -m unittest tests/test_buffer.py

```

on cluster 

```bash
sbatch run.sbatch configs/ppo.yaml

sbatch run.sbatch configs/mb_mpc.yaml

```

meta_learned_bias_term_adaptation

meta_learned_residual_adaptation

meta_learned_low_rank_adaptation