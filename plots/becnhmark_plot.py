import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# List of CSV files to compare
benchmark_files = [
    "arap_benchmark_PAPER_ARAP_CHOLESKY.csv",
    "arap_benchmark_PROJECTED_ARAP_CHOLESKY.csv"
]

# Base directory
benchmark_dir = "benchmark_data"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Helper to extract label from file name
def extract_label(filename):
    match = re.match(r"arap_benchmark_(.*?)_ARAP_(.*?)\.csv", filename)
    if match:
        impl, solver = match.groups()
        return f"{impl} + {solver}"
    return filename

# Timing Plot
plt.figure(figsize=(10, 6))
for file in benchmark_files:
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    plt.plot(df["Iterations"], df["Time(us)"], marker='o', label=label)

plt.title("ARAP Solver Benchmark - Timing")
plt.xlabel("Iterations")
plt.ylabel("Time (µs)")
plt.grid(True)
plt.legend()
timing_output = os.path.join(plot_dir, "arap_benchmark_timing_plot.png")
plt.savefig(timing_output)
print(f"Timing plot saved to {timing_output}")

# Convergence Plot (ΔV)
plt.figure(figsize=(10, 6))
for file in benchmark_files:
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)

    # Skip the first row for convergence plot
    df_convergence = df.iloc[1:]
    plt.plot(df_convergence["Iterations"], df_convergence["VertexChangeNorm"], marker='s', label=label)


plt.title("ARAP Solver Benchmark - Convergence (ΔV)")
plt.xlabel("Iterations")
plt.ylabel("Vertex Change Norm (ΔV)")
plt.grid(True)
plt.legend()
convergence_output = os.path.join(plot_dir, "arap_benchmark_convergence_plot.png")
plt.savefig(convergence_output)
print(f"Convergence plot saved to {convergence_output}")

plt.figure(figsize=(10, 6))
for file in benchmark_files:
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)

    plt.plot(df["Iterations"], df["TargetError"], marker='^', label=label)

plt.title("ARAP Solver Benchmark - Error to Target")
plt.xlabel("Iterations")
plt.ylabel("Target Error (L2 Norm)")
plt.grid(True)
plt.legend()
target_error_output = os.path.join(plot_dir, "arap_benchmark_target_error_plot.png")
plt.savefig(target_error_output)
print(f"Target error plot saved to {target_error_output}")
