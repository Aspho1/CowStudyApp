from skopt.plots import plot_convergence
from skopt import load
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# def make_compound_plot(path1, path2):
#     """Save optimization visualization plots"""
#     r1 = load(path1)
#     r2 = load(path2)

#     print(type(r1)) #<class 'scipy.optimize._optimize.OptimizeResult'>

#     _, ax = plt.subplots(figsize=(10, 6), layout='constrained')

#     plot_convergence(
#             [r1,r2],
#             ax=ax
#         )

#     plt.show()
#     # plt.savefig('data/cv_results/RB_22/LSTM/double_conv.png', dpi=300)
#     # plt.close()

# make_compound_plot(
#     path1='data/cv_results/RB_22/LSTM/ops/v3job/bayes_opt_results.pkl',
#     path2='data/cv_results/RB_22/LSTM/opo/v2job/bayes_opt_results.pkl'
# )


def make_compound_plot(path1, path2):
    """Save optimization visualization plots"""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    width_in_inches = 190/25.4
    height_in_inches = width_in_inches * 0.5  # Slightly shorter to reduce white space

    r1 = load(path1)
    r2 = load(path2)

    # Debug information
    print(f"r1 func_vals shape: {np.shape(r1.func_vals)}")
    print(f"r2 func_vals shape: {np.shape(r2.func_vals)}")
    
    # Create a single plot with custom plotting
    plt.figure(layout='constrained', figsize=(width_in_inches, height_in_inches))
    
    # Plot the convergence manually
    xs1 = range(1, len(r1.func_vals) + 1)
    xs2 = range(1, len(r2.func_vals) + 1)
    
    # Plot minimum observed value at each iteration
    plt.plot(xs1, np.minimum.accumulate(r1.func_vals), label="One per sequence")
    plt.plot(xs2, np.minimum.accumulate(r2.func_vals), label="One per observation")
    

    plt.ylim(-1,-0.75)
    plt.xlabel("Number of calls $n$")
    plt.ylabel("min $-f(x)$")
    plt.legend()
    # plt.title("Convergence plot")
    plt.savefig('data/cv_results/RB_22/LSTM/double_conv.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

make_compound_plot(
    path1='data/cv_results/RB_22/LSTM/ops/v2/bayes_opt_results.pkl',
    path2='data/cv_results/RB_22/LSTM/opo/v2/bayes_opt_results.pkl'
)
