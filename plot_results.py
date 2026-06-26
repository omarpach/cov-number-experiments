import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load and Prep the Data ---
df = pd.read_csv("results/experiment_log_v3.csv")

df = df.sort_values(by="eta")

# --- 2. Create the Plot ---
plt.figure(figsize=(10, 7))

# --- 3. Group and Plot ---
# Group by dataset size, seed, AND algorithm
for (n_points, seed, algo), group_data in df.groupby(
    ["num_points", "seed", "algorithm"]
):
    plt.plot(
        group_data["eta"],
        group_data["covering_number"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        label=f"{algo} (N={n_points:,})",
    )

# --- 4. Formatting ---
plt.title("log $\\eta$-Cover Size vs. Radius ($\\eta$)", fontsize=14, fontweight="bold")
plt.xlabel("Radius $\\eta$", fontsize=12)
plt.ylabel("log Covering Number $|Q|$", fontsize=12)

# Move the legend outside the plot if it gets too crowded with 3 algorithms x 3 sizes
plt.legend(
    title="Algorithm & Dataset Size",
    fontsize=10,
    title_fontsize=11,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.yscale("log")

plt.tight_layout()
plt.show()
