import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Function to test multiple distributions and print results
def test_distributions(data, distributions):
    results = []

    for dist_name, distribution in distributions.items():
        # Fit the distribution to the data
        params = distribution.fit(data)

        # Perform KS test
        ks_statistic, p_value = stats.kstest(data, distribution.name, args=params)

        results.append({
            'Distribution': dist_name,
            'KS Statistic': ks_statistic,
            'P-value': p_value,
            'Parameters': params
        })

    # Convert to DataFrame for nice display
    return pd.DataFrame(results).sort_values('KS Statistic')


# Visual check - Create QQ plots for the best-fitting distributions
def create_qq_plots(data, results, title_prefix, n_best=3):
    plt.figure(figsize=(15, 5))
    best_dists = results.head(n_best)

    for i, (_, row) in enumerate(best_dists.iterrows()):
        plt.subplot(1, n_best, i+1)
        dist_name = row['Distribution']
        distribution = distributions[dist_name]
        params = row['Parameters']

        # Generate theoretical quantiles
        theoretical_quantiles = distribution.ppf(np.linspace(0.01, 0.99, 100), *params[:-2], loc=params[-2], scale=params[-1])

        # Get sample quantiles
        sample_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))

        # Plot
        plt.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5)
        plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
                 [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
        plt.title(f"{dist_name}\nKS={row['KS Statistic']:.4f}, p={row['P-value']:.4e}")
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')

    plt.suptitle(f"{title_prefix} - QQ Plots for Best Fitting Distributions")
    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_')}_qq_plots.png")
    plt.show()


# Create histograms with fitted distributions
def plot_histogram_with_fits(data, results, title_prefix, n_best=3):
    plt.figure(figsize=(15, 5))
    best_dists = results.head(n_best)

    # Create bins for histogram
    counts, bins = np.histogram(data, bins=100, density=True)

    plt.subplot(1, 1, 1)
    plt.hist(data, bins=100, density=True, alpha=0.5, label='Data')

    # Plot PDF for best distributions
    x = np.linspace(min(data), max(data), 1000)
    for _, row in best_dists.iterrows():
        dist_name = row['Distribution']
        distribution = distributions[dist_name]
        params = row['Parameters']

        pdf = distribution.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
        plt.plot(x, pdf, label=f"{dist_name} (KS={row['KS Statistic']:.4f})")

    plt.title(f"{title_prefix} - Histogram with Fitted Distributions")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_')}_histogram_fits.png")
    plt.show()


# Create visualization of ECDF vs fitted CDFs
def plot_ecdf_with_fits(data, results, title_prefix, n_best=3):
    plt.figure(figsize=(15, 5))
    best_dists = results.head(n_best)

    # Calculate ECDF
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    plt.subplot(1, 1, 1)
    plt.plot(x, y, 'o', markersize=2, label='Empirical CDF')

    # Plot theoretical CDFs
    x_smooth = np.linspace(min(data), max(data), 1000)
    for _, row in best_dists.iterrows():
        dist_name = row['Distribution']
        distribution = distributions[dist_name]
        params = row['Parameters']

        cdf = distribution.cdf(x_smooth, *params[:-2], loc=params[-2], scale=params[-1])
        plt.plot(x_smooth, cdf, label=f"{dist_name} (KS={row['KS Statistic']:.4f})")

    plt.title(f"{title_prefix} - ECDF with Fitted Distributions")
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title_prefix.replace(' ', '_')}_ecdf_fits.png")
    plt.show()


data = pd.read_csv("predictions.csv")

data['log_mag_var'] = np.log(data['magnitude_var']+1).dropna()
# data['log_mag_var'] = data[data['log_mag_var']<6]

labels = ["All", "Labeled", "Grazing", "Resting", "Traveling"]
datasets = [
     data
    ,data[data["activity"] != "NA"]
    ,data[data["activity"] == "Grazing"]
    ,data[data["activity"] == "Resting"]
    ,data[data["activity"] == "Traveling"]
    ]

for l, d in zip(labels,datasets):
    log_mag_var = d['log_mag_var']

    print(f"\n\n{'-'*80} {l} {'-'*80}")
    print(f"\nLog-Transformed Data Statistics:")
    print(log_mag_var.describe())
    # Distributions to test
    distributions = {
        'Normal': stats.norm,
        'Log-normal': stats.lognorm,
        'Exponential': stats.expon,
        'Gamma': stats.gamma,
        'Weibull': stats.weibull_min,
        'Pareto': stats.pareto
    }

    # Test distributions on log-transformed data
    print("\nKS Test Results for Log-Transformed Data :")
    log_results = test_distributions(log_mag_var, distributions)
    print(log_results[['Distribution', 'KS Statistic', 'P-value']])


    # Generate visualizations
    # create_qq_plots(mag_var, original_results, "Original Data")
    # create_qq_plots(log_mag_var, log_results, "Log-Transformed Data")

    # plot_histogram_with_fits(mag_var, original_results, "Original Data")
    # plot_histogram_with_fits(log_mag_var, log_results, "Log-Transformed Data")

    # # plot_ecdf_with_fits(mag_var, original_results, "Original Data")
    plot_ecdf_with_fits(log_mag_var, log_results, "Log-Transformed Data")
