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

976119 a100q l2ra ismailou PD 0:00 1 (ReqNodeNotAvail, Reserved for maintenance)
974776 dgx2q l2ra ismailou PD 0:00 1 (ReqNodeNotAvail, Reserved for maintenance)