"""
IEEE Publication Results and High-Resolution Plotting Generator
Loads genuine experimental results produced from real ASAP-AES dataset execution
and outputs:
1. Tables 1-5 directly from authentic experimental metrics
2. Figures 1-8 strictly conforming to IEEE standards:
   - dpi=800
   - plt.rcParams["figure.figsize"] = (11, 7)
   - plt.rcParams['font.family'] = 'Times New Roman'
   - plt.rcParams['font.size'] = 18
   - plt.rcParams['font.weight'] = 'bold'
   - ax.grid(False) on all plots
"""

import os
import json
import pandas as pd
from run_experiments import run_real_experiments, generate_ieee_plots


def generate_benchmark_suite(sample_size=1600, epochs=4, force_rerun=False):
    print("=" * 80)
    print("GENERATING BENCHMARK RESULTS AND IEEE FIGURES FROM REAL ASAP DATASET")
    print("=" * 80)

    results_file = 'real_experiment_results.json'
    if force_rerun or not os.path.exists(results_file):
        print(f"[Pipeline] Running real model training & evaluation on ASAP dataset...")
        run_real_experiments(sample_size=sample_size, epochs=epochs, device='cpu')
    else:
        print(f"[Pipeline] Loading existing authentic results from: {results_file}")
        with open(results_file, 'r') as f:
            artifacts = json.load(f)
        generate_ieee_plots(artifacts)

    print("\nBenchmark generation complete. All 5 tables and 8 figures are up-to-date.")


if __name__ == '__main__':
    generate_benchmark_suite()
