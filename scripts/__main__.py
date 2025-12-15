"""
Entry point for running scripts as a module.

`python -m scripts` will default to training both HalfCheetah variants via PPO.
"""
from scripts.train_both_half_cheetahs import main


if __name__ == "__main__":
    main()
