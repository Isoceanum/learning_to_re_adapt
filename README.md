# learning_to_re_adapt


```bash
python -m scripts.train configs/current.yaml


python -m scripts.render outputs/2025-09-03/hopper_ppo_test1


python -m scripts.evaluate_run outputs/2025-09-03/hopper_ppo_test1


python -m scripts.test_perturbations.py configs/mb_mpc_cheetah_perturbation.yaml --episodes 5
