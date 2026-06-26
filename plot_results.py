import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load and Prep the Data ---
# Point this to your new CSV file
df = pd.read_csv("results/experiment_log_v2.csv")

# Sort by eta so the lines draw cleanly from left to right
df = df.sort_values(by="eta")

# --- 2. Create the Plot ---
plt.figure(figsize=(9, 6))

# --- 3. Group and Plot ---
# NEW: Group by BOTH the number of points and the seed.
# This ensures each unique run gets its own clean, continuous line.
for (n_points, seed), group_data in df.groupby(["num_points", "seed"]):
    plt.plot(
        group_data["eta"],
        group_data["covering_number"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label=f"$N = {n_points:,}$ (Seed: {seed})",
    )

# --- 4. Formatting ---
plt.title("log $\\eta$-Cover Size vs. Radius ($\\eta$)", fontsize=14, fontweight="bold")
plt.xlabel("Radius $\\eta$", fontsize=12)
plt.ylabel("log Covering Number $|Q|$", fontsize=12)

# Update the legend title
plt.legend(title="Dataset Size & Seed", fontsize=11, title_fontsize=12)

plt.grid(True, which="both", linestyle="--", alpha=0.6)

# Leave the log scale enabled since your covers will range from 1 to 20,000
plt.yscale("log")

plt.tight_layout()
plt.show()
