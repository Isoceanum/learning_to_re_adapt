import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

# Folder that contains your CSV files (change if needed)
data_dir = "csv"

# Find all CSV files in that folder
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

plt.figure()

for path in csv_files:
    # Read one csv
    df = pd.read_csv(path)

    # Use filename (without .csv) as the curve label
    label = os.path.splitext(os.path.basename(path))[0]

    # Plot time_steps (x) vs reward_mean (y)
    plt.plot(df["time_steps"], df["reward_mean"], label=label)

# Make the plot look nice
plt.xlabel("Time steps")
plt.ylabel("Average return (reward_mean)")
plt.title("Reward vs Time steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
plt.show()

