import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv("times.csv")

# Rename for convenience
df = df.rename(columns={"time[s]": "time"})

# Sort by processor count (important for plots)
df = df.sort_values("processors")

# ==== Compute metrics ====

# Strong scaling baseline: smallest processor count
p_min = df["processors"].min()
t_min = df.loc[df["processors"] == p_min, "time"].iloc[0]

df["speedup"] = t_min / df["time"]
df["efficiency"] = df["speedup"] / df["processors"]

# Amdahl serial fraction:
# f = (p/S - 1) / (p - 1)
df["serial_fraction"] = (df["processors"] / df["speedup"] - 1) / (df["processors"] - 1)

os.makedirs("plots", exist_ok=True)

# ==== 1. Runtime ====
plt.figure(figsize=(6, 4))
plt.plot(df["processors"], df["time"], marker="o")
plt.xlabel("Processors")
plt.ylabel("Time (s)")
plt.title("Runtime vs Processors")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/runtime_vs_processors.png")

# ==== 2. Speedup (scaling) ====
plt.figure(figsize=(6, 4))
plt.plot(df["processors"], df["speedup"], marker="o", label="Measured")

# Ideal linear scaling
plt.plot(
    df["processors"], df["processors"] / df["processors"].min(), "--", label="Ideal"
)

plt.xlabel("Processors")
plt.ylabel("Speedup")
plt.title("Strong Scaling Speedup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/speedup_vs_processors.png")

# ==== 3. Efficiency ====
plt.figure(figsize=(6, 4))
plt.plot(df["processors"], df["efficiency"], marker="o")
plt.xlabel("Processors")
plt.ylabel("Efficiency")
plt.title("Parallel Efficiency")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/efficiency_vs_processors.png")

# ==== 4. Serial fraction ====
plt.figure(figsize=(6, 4))
plt.plot(df["processors"], df["serial_fraction"], marker="o")
plt.xlabel("Processors")
plt.ylabel("Serial Fraction")
plt.title("Serial Fraction (Amdahl)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/serial_fraction_vs_processors.png")
