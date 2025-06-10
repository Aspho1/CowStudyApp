import numpy as np
from skopt import load

import matplotlib
print("Matplotlib configuration directory:", matplotlib.get_configdir())
print("Matplotlib configuration file:", matplotlib.matplotlib_fname())

matplotlib.use('QtAgg')

from matplotlib import pyplot as plt

from pathlib import Path


import pandas as pd
import seaborn as sns

mode = 'opo'
# Path to your optimization results file - you can edit this directly
pkl_path = f"data/cv_results/RB_22/LSTM/{mode}/v2job/bayes_opt_results.pkl"

# Load the optimization results
result = load(pkl_path)
input_vals = result.x_iters
func_vals = result.func_vals
srt_idx = np.argsort(func_vals)

print(f"Reading optimization results from: {pkl_path}")
def fix_type(value, spacing):
    # print(value, type(value))
    if isinstance(value, (np.integer, np.int32, np.int64, int)):
        return f"{int(value):<{spacing}}"
    elif isinstance(value, (np.floating, np.float32, np.float64, float)):
        return f"{float(value):<{spacing}.{7}f}"
    else:
        return value

lbls = [
    "idx",
    "max_length",
    "batch_size",
    "initial_lr",
    "decay_steps",
    "decay_rate",
    "clipnorm ",
    "lstm_size",
    "dense_size"



    # "patience",
    # "min_delta ",
    # "reg_val"
]

lens = [len(i) for i in lbls]

print(len(func_vals))

# print(result)
print(" | ".join([x for x in lbls]))
for i in srt_idx[:10]:
    if i == (len(input_vals) - 1):
        continue
    try:
        print(f"{i:<{lens[0]}} | {' | '.join([fix_type(x,lens[j+1]) for j, x in enumerate(input_vals[i])])} --> {-100*func_vals[i]:>0.2f}")
    except:
        print("Failure on i =",i)



# Create scatter plots for each parameter vs objective function
param_names = lbls[1:]  # Skip 'idx'
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# For better visualization, convert func_vals to percentages (similar to your print)
perf_vals = -100 * func_vals

# Extract input values for each parameter
param_values = []
for j in range(len(param_names)):
    param_values.append([row[j] for row in input_vals])

# Create scatter plots
for i, (ax, param_name, values) in enumerate(zip(axes, param_names, param_values)):
    ax.scatter(values, perf_vals, alpha=0.6)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Performance (%)')
    ax.set_title(f'{param_name} vs Performance')
    
    # Add trendline
    if len(values) > 1:  # Make sure we have enough points
        try:
            z = np.polyfit(values, perf_vals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(values), p(sorted(values)), "r--", alpha=0.8)
        except:
            # Some parameters might not be numeric or cause fitting issues
            pass

plt.subplots_adjust(
    top=0.96,
    bottom=0.064,
    left=0.071,
    right=0.966,
    hspace=0.42,
    wspace=0.323
)
plt.savefig(Path(pkl_path).parent / f'parameter_performance_plots_{mode}.png')

plt.show()

# Create a DataFrame with all parameters and performance
data = {param_name: values for param_name, values in zip(param_names, param_values)}
data['performance'] = perf_vals
df = pd.DataFrame(data)

# Calculate correlation matrix
corr = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Parameter Correlation Matrix')
plt.tight_layout()
plt.savefig(Path(pkl_path).parent / f'parameter_correlation_heatmap_{mode}.png')
plt.show()

# readingpkl.py:122: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()
