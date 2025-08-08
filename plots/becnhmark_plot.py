import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# List of CSV files to compare - all available solver configurations
benchmark_files = [
    "arap_benchmark_PAPER_ARAP_PAPER_CHOLESKY.csv",
    "arap_benchmark_PAPER_ARAP_PAPER_LDLT.csv",
    "arap_benchmark_PAPER_ARAP_PAPER_LU.csv",
    "arap_benchmark_CERES_ARAP_CHOLESKY.csv", 
    "arap_benchmark_CERES_ARAP_SPARSE_SCHUR.csv",
    # "arap_benchmark_CERES_ARAP_CGNR.csv",
    "arap_benchmark_IGL_ARAP_CHOLESKY.csv"
]

# Base directory
mesh_name = "cactus_highres"
benchmark_dir = f"benchmark_data/{mesh_name}"
plot_dir = f"plots/{mesh_name}"
os.makedirs(plot_dir, exist_ok=True)

# Helper to extract label from file name
def extract_label(filename):
    # Handle the new naming convention from the updated benchmark
    if "PAPER_ARAP_PAPER_CHOLESKY" in filename:
        return "Paper ARAP (Cholesky)"
    elif "PAPER_ARAP_PAPER_LDLT" in filename:
        return "Paper ARAP (LDLT)"
    elif "PAPER_ARAP_PAPER_LU" in filename:
        return "Paper ARAP (LU)"
    elif "CERES_ARAP_CHOLESKY" in filename:
        return "Ceres ARAP (Cholesky)"
    elif "CERES_ARAP_SPARSE_SCHUR" in filename:
        return "Ceres ARAP (Sparse Schur)"
    elif "CERES_ARAP_CGNR" in filename:
        return "Ceres ARAP (CGNR)"
    elif "IGL_ARAP_CHOLESKY" in filename:
        return "IGL ARAP (Cholesky)"
    else:
        # Fallback to original pattern matching
        match = re.match(r"arap_benchmark_(.*?)_ARAP_(.*?)\.csv", filename)
        if match:
            impl, solver = match.groups()
            return f"{impl} + {solver}"
        return filename

# Define colors and markers for consistent visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'v', 'D', 'x']

# Timing Plot
plt.figure(figsize=(12, 8))
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    # Convert microseconds to milliseconds
    plt.plot(df["Iterations"], df["Time(us)"] / 1000, marker=marker, color=color, label=label, linewidth=2, markersize=6)

plt.title("ARAP Solver Benchmark - Timing Comparison", fontsize=22, fontweight='bold')
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Time (ms)", fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
timing_output = os.path.join(plot_dir, "arap_benchmark_timing_plot.png")
plt.savefig(timing_output, dpi=300, bbox_inches='tight')
print(f"Timing plot saved to {timing_output}")
plt.close()

# Convergence Plot (ΔV)
plt.figure(figsize=(12, 8))
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    # Skip the first row for convergence plot (first iteration has ΔV = 0)
    df_convergence = df.iloc[1:]
    plt.plot(df_convergence["Iterations"], df_convergence["VertexChangeNorm"], 
             marker=marker, color=color, label=label, linewidth=2, markersize=6)

plt.title("ARAP Solver Benchmark - Convergence Rate (ΔV)", fontsize=22, fontweight='bold')
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Vertex Change RMSE (ΔV)", fontsize=18)
plt.yscale('log')  # Log scale for better visualization of convergence
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
convergence_output = os.path.join(plot_dir, "arap_benchmark_convergence_plot.png")
plt.savefig(convergence_output, dpi=300, bbox_inches='tight')
print(f"Convergence plot saved to {convergence_output}")
plt.close()

# Target Error Plot
plt.figure(figsize=(12, 8))
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    plt.plot(df["Iterations"], df["TargetError"], marker=marker, color=color, label=label, linewidth=2, markersize=6)

plt.title("ARAP Solver Benchmark - Accuracy vs Reference Solution", fontsize=22, fontweight='bold')
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Target Error (RMSE)", fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
target_error_output = os.path.join(plot_dir, "arap_benchmark_target_error_plot.png")
plt.savefig(target_error_output, dpi=300, bbox_inches='tight')
print(f"Target error plot saved to {target_error_output}")
plt.close()

# Edge Length RMSE Plot
plt.figure(figsize=(12, 8))
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(df["Iterations"], df["EdgeLengthRelRMSError"], marker=marker, color=color, label=label, linewidth=2, markersize=6)

plt.title("ARAP Solver Benchmark - Shape Preservation (Edge Length Error)", fontsize=22, fontweight='bold')
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Relative Edge Length RMSE", fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
edge_error_output = os.path.join(plot_dir, "arap_benchmark_edge_rmse_plot.png")
plt.savefig(edge_error_output, dpi=300, bbox_inches='tight')
print(f"Edge length RMSE plot saved to {edge_error_output}")
plt.close()

# Create a comprehensive comparison plot with subplots in a row
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
plt.subplots_adjust(wspace=0.3)  # Add space between subplots to keep them square

# Timing subplot
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    # Convert microseconds to milliseconds
    ax1.plot(df["Iterations"], df["Time(us)"] / 1000, marker=marker, color=color, label=label, linewidth=2, markersize=4)

ax1.set_title("Timing Performance", fontweight='bold', fontsize=18)
ax1.set_xlabel("Iterations", fontsize=16)
ax1.set_ylabel("Time (ms)", fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=14)

# Convergence subplot (log scale)
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    df_convergence = df.iloc[1:]  # Skip first iteration
    ax2.plot(df_convergence["Iterations"], df_convergence["VertexChangeNorm"], 
             marker=marker, color=color, label=label, linewidth=2, markersize=4)

ax2.set_title("Convergence Rate", fontweight='bold', fontsize=18)
ax2.set_xlabel("Iterations", fontsize=16)
ax2.set_ylabel("Vertex Change RMSE (ΔV)", fontsize=16)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Target Error subplot
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    ax3.plot(df["Iterations"], df["TargetError"], marker=marker, color=color, label=label, linewidth=2, markersize=4)

ax3.set_title("Accuracy vs Reference", fontweight='bold', fontsize=18)
ax3.set_xlabel("Iterations", fontsize=16)
ax3.set_ylabel("Target Error (RMSE)", fontsize=16)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=14)

# Edge preservation subplot
for i, file in enumerate(benchmark_files):
    path = os.path.join(benchmark_dir, file)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    label = extract_label(file)
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    ax4.plot(df["Iterations"], df["EdgeLengthRelRMSError"], marker=marker, color=color, label=label, linewidth=2, markersize=4)

ax4.set_title("Shape Preservation", fontweight='bold', fontsize=18)
ax4.set_xlabel("Iterations", fontsize=16)
ax4.set_ylabel("Relative Edge Length RMSE", fontsize=16)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='both', which='major', labelsize=14)

# Create a single horizontal legend below all subplots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
           ncol=len(labels), fontsize=18, frameon=False)

# plt.suptitle("ARAP Solver Comprehensive Comparison", fontsize=20, fontweight='bold')
plt.suptitle("", fontsize=24, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the legend below
comprehensive_output = os.path.join(plot_dir, "arap_benchmark_comprehensive_comparison.png")
plt.savefig(comprehensive_output, dpi=300, bbox_inches='tight')
print(f"Comprehensive comparison plot saved to {comprehensive_output}")
plt.close()

print("\n" + "="*60)
print("BENCHMARK PLOTTING COMPLETED!")
print("="*60)
print(f"All plots saved to: {plot_dir}")
print("Generated plots:")
print("  1. Timing comparison")
print("  2. Convergence rate comparison")
print("  3. Accuracy vs reference solution")
print("  4. Shape preservation comparison")
print("  5. Comprehensive 4-in-1 comparison")
print("="*60)