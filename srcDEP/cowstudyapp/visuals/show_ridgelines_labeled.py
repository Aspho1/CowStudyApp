import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Define features to analyze
features = ['step', 'magnitude_mean', 'magnitude_var']

# Define valid activity states
valid_states = ["Grazing", "Resting", "Traveling"]

# Load data
# Modify this path to match your environment
target_dataset = pd.read_csv("data/analysis_results/hmm/FinalPredictions/RB_Paper_Model_Preds_logmagvar/predictions.csv")

x_ticks = {
    'step': [0, 50, 100, 150, 200, 250, 300], 
    'magnitude_mean': [6, 7, 8, 9, 10, 11, 12], 
    'magnitude_var': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
}

# Define feature labels with units
feature_labels = {
    'step': 'Step Size (m)',
    'magnitude_mean': 'MeanSVM (m/s²)',
    'magnitude_var': 'VarSVM ln(1+(m²/s⁴))'
}

# Define custom bandwidths for each feature
feature_bandwidths = {
    'step': 12,
    'magnitude_mean': 0.3,
    'magnitude_var': 0.2    
}

# Define colors for activities
activity_colors = {
    "Resting": "#4e79a7",   # Blue
    "Grazing": "#f28e2b",    # Orange
    "Traveling": "#59a14f"   # Green
}

# Filter data and select columns
filtered_data = target_dataset[target_dataset['activity'].isin(valid_states)][
    ['ID', 'activity', 'step', 'magnitude_mean', 'magnitude_var', 'predicted_state']
]

# Reshape the data from wide to long format for faceting
plot_data = pd.melt(
    filtered_data, 
    id_vars=['ID', 'activity', 'predicted_state'], 
    value_vars=features,
    var_name='feature',
    value_name='value'
)

# Remove extreme outliers per feature (keep up to 99th percentile)
# Fix the deprecation warning by including include_groups=False
plot_data = plot_data.groupby('feature', as_index=False, group_keys=False).apply(
    lambda x: x.assign(
        value=np.minimum(x['value'], x['value'].quantile(0.99))
    )
)

# Calculate accuracy by cow and activity
accuracy_by_cow_activity = filtered_data.groupby(['ID', 'activity'], as_index=False, group_keys=False).apply(
    lambda x: pd.Series({
        'accuracy': (x['predicted_state'] == x['activity']).mean()
    })
).reset_index()

# Join the accuracy information with the plot data
plot_data = pd.merge(
    plot_data, 
    accuracy_by_cow_activity, 
    on=['ID', 'activity'],
    how='left'
).sort_values("ID")

cow_ids = plot_data['ID'].unique()
id_to_position = {id: i for i, id in enumerate(cow_ids)}
plot_data['y_position'] = plot_data['ID'].map(id_to_position)

# Set font sizes for better readability
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
ID_SIZE = 10  # Slightly smaller for cow IDs since there are many

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TICK_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': ID_SIZE,
    'legend.fontsize': LABEL_SIZE,
    'figure.titlesize': TITLE_SIZE
})



width_in_inches = 190/25.4
height_in_inches = width_in_inches * (5/4)

# Create a single combined figure 
fig, axs = plt.subplots(3, 3, figsize=(width_in_inches, height_in_inches), layout='constrained')

for act_idx, activity_label in enumerate(valid_states):
    ax1 = axs[act_idx]
    activity_data = plot_data[plot_data['activity'] == activity_label]

    for fea_idx, feature_label in enumerate(features):
        xtick = x_ticks[feature_label]
        ax = ax1[fea_idx]

        feature_data = activity_data[activity_data['feature'] == feature_label]
 
        # Create a custom 1D KDE for each cow ID
        for cow_id in cow_ids:
            cow_data = feature_data[feature_data['ID'] == cow_id]
            if len(cow_data) > 2:  # Need at least 3 points for KDE
                y_position = id_to_position[cow_id]
                accuracy = min(1.0, cow_data['accuracy'].iloc[0])
                
                # Generate x values for the KDE
                x_range = np.linspace(x_ticks[feature_label][0], 
                                      x_ticks[feature_label][-1], 
                                      1000)
                
                # Calculate KDE directly without try/except
                bandwidth = feature_bandwidths[feature_label]/cow_data['value'].std(ddof=1)
                kde = stats.gaussian_kde(cow_data['value'].dropna(), bw_method=bandwidth)
                y_values = kde(x_range)
                
                # Scale the density to a reasonable height
                y_values = y_values / np.max(y_values) * 0.8
                
                # Fill the area under the curve
                ax.fill_between(x_range, 
                                y_position, 
                                y_position + y_values, 
                                alpha=min(1.0,0.1 + (max(0.5,accuracy) - 0.5) / 0.5),  # Use accuracy for transparency
                                color=activity_colors[activity_label])
                
                # Add a thin outline for better visibility
                ax.plot(x_range, y_position + y_values, 
                        color=activity_colors[activity_label], 
                        linewidth=0.5, 
                        alpha=1
                        )

        # First Column
        if fea_idx == 0:
            ax.set_yticks(range(len(cow_ids)), cow_ids)
            ax.set_ylabel(f"{activity_label}\nCow ID", fontsize=LABEL_SIZE)
            # Make sure the cow IDs are clearly visible
            ax.tick_params(axis='y', labelsize=ID_SIZE)
        else:
            ax.set_yticks(range(len(cow_ids)), [])
            ax.set_ylabel("")
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False)
        
        # Bottom Row
        if act_idx == len(valid_states)-1:
            ax.set_xticks(xtick)
            ax.set_xticklabels(xtick, fontsize=TICK_SIZE)
            ax.set_xlabel(feature_labels[feature_label], fontsize=LABEL_SIZE)
        else:
            ax.set_xticks(xtick)
            ax.set_xticklabels([])
            ax.set_xlabel("")
            # ax.spines['bottom'].set_visible(False)

            ax.tick_params(axis='x', which='both', bottom=False)
        
        # Clean up borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid lines
        ax.grid(True, axis='y', linestyle='-', alpha=0.1)

        ax.set_xlim(xtick[0], xtick[-1])
        ax.set_ylim(0, len(cow_ids))

# # Add column titles
# for i, feature in enumerate(features):
#     axs[0, i].set_title(feature_labels[feature], fontsize=TITLE_SIZE)

plt.savefig("src/cowstudyapp/descriptive_stats/cow_behavior_features.png", dpi=300, bbox_inches='tight')
plt.show()