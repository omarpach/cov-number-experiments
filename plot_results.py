import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load and Prep the Data ---
df = pd.read_csv("results/experiment_log.csv")

# Sort by eta so the lines draw cleanly from left to right
df = df.sort_values(by="eta")

# --- 2. Create the Plot ---
plt.figure(figsize=(9, 6))

# --- 3. Group and Plot ---
# df.groupby("num_points") automatically splits your dataframe into separate chunks
# based on the unique values in the 'num_points' column (your n and n' experiments)
for n_points, group_data in df.groupby("num_points"):
    plt.plot(
        group_data["eta"],
        group_data["covering_number"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label=f"$N = {n_points:,}$",  # Adds commas to large numbers in the legend
    )

# --- 4. Formatting ---
plt.title("log $\\eta$-Cover Size vs. Radius ($\\eta$)", fontsize=14, fontweight="bold")
plt.xlabel("Radius $\\eta$", fontsize=12)
plt.ylabel("log Covering Number $|Q|$", fontsize=12)

# Add the legend to differentiate the two lines
plt.legend(title="Dataset Size", fontsize=11, title_fontsize=12)

plt.grid(True, which="both", linestyle="--", alpha=0.6)

# Uncomment this if the covering numbers drop off too steeply to see the details
plt.yscale("log")

plt.tight_layout()
plt.show()
