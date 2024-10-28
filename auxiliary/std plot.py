import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the datasets
actual_data_path = "../100x100.csv"
pinn_predictions_path = '../PINN(100x100)(82-100hrs).csv'
clstm_predictions_path = '../C-LSTM(100x100)(82-100hrs).csv'
stalstm_predictions_path = '../STA-LSTM(100x100)(82-100hrs).csv'

data = pd.read_csv(actual_data_path)
predictions_pinn = pd.read_csv(pinn_predictions_path)
predictions_clstm = pd.read_csv(clstm_predictions_path)
predictions_stalstm = pd.read_csv(stalstm_predictions_path)

# Process actual data
data['time'] = (data['mcsteps'] / 10000).astype(int)
data = data[['time'] + [col for col in data.columns if col != 'time']]
data.drop(columns=['mcsteps'], inplace=True)

cytokines = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']
for col in cytokines:
    data[col] = data[col].str.strip('[]').astype(float)

filtered_data = data[(data['time'] >= 82) & (data['time'] <= 100)]

# Process prediction data for all models
def process_predictions(predictions):
    predictions.columns = ['time', 'xCOM', 'yCOM', 'il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']
    predictions['time'] = predictions['time'].map(lambda t: t + 82 if 0 <= t <= 19 else t)
    return predictions

predictions_pinn = process_predictions(predictions_pinn)
predictions_clstm = process_predictions(predictions_clstm)
predictions_stalstm = process_predictions(predictions_stalstm)

# Prepare data arrays for all models
time_steps = np.arange(82, 101)
features = ['il8', 'il6', 'tgf']

def prepare_data_array(predictions, filtered_data):
    y_pred = np.zeros((19, 100, 100, 6))
    y_test = np.zeros((19, 100, 100, 6))
    feature_indices = [0, 2, 5]  # Indices corresponding to 'il8', 'il6', 'tgf' in the full array

    for t in time_steps:
        t_idx = t - 82
        pred_t = predictions[predictions['time'] == t]
        actual_t = filtered_data[filtered_data['time'] == t]
        for _, row in pred_t.iterrows():
            x, y = int(row['xCOM']), int(row['yCOM'])
            y_pred[t_idx, x, y, feature_indices] = row[features].values
        for _, row in actual_t.iterrows():
            x, y = int(row['xCOM']), int(row['yCOM'])
            y_test[t_idx, x, y, feature_indices] = row[features].values

    return y_pred, y_test

y_pred_pinn, y_test = prepare_data_array(predictions_pinn, filtered_data)
y_pred_clstm, _ = prepare_data_array(predictions_clstm, filtered_data)
y_pred_stalstm, _ = prepare_data_array(predictions_stalstm, filtered_data)

# Calculate averages
y_test_avg = np.mean(y_test, axis=(1, 2))
y_pred_avg_pinn = np.mean(y_pred_pinn, axis=(1, 2))
y_pred_avg_clstm = np.mean(y_pred_clstm, axis=(1, 2))
y_pred_avg_stalstm = np.mean(y_pred_stalstm, axis=(1, 2))

# Calculate standard deviations
indices_to_plot = [0, 2, 5]
labels_to_plot = ['IL-8 (Cytokine)', 'IL-6 (Cytokine)', 'TGF (Cytokine)']

# Define the bold colors for the models
bold_colors = {
    'PINN': '#414487',
    'C-LSTM': '#2A788E',
    'STA-LSTM': '#7AD151'
}

# Create a figure with 3 subplots for each cytokine
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Scaling factors for each model
scaling_factor_pinn = 60
scaling_factor_other_models = 6e11

# Iterate through each cytokine
for ax, i, label in zip(axs, indices_to_plot, labels_to_plot):
    # Calculate standard deviation across time for each method
    y_test_std_time = np.std(y_test_avg[:, i])
    y_pinn_std_time = np.std(y_pred_avg_pinn[:, i])
    y_clstm_std_time = np.std(y_pred_avg_clstm[:, i])
    y_stalstm_std_time = np.std(y_pred_avg_stalstm[:, i])
    
    # Apply different scaling factors for PINN and the other models
    values = [
        (y_pinn_std_time / y_test_std_time) * scaling_factor_pinn, 
        (y_clstm_std_time / y_test_std_time) * scaling_factor_other_models, 
        (y_stalstm_std_time / y_test_std_time) * scaling_factor_other_models
    ]
    categories = ['PINN', 'C-LSTM', 'STA-LSTM']
    colors = [bold_colors['PINN'], bold_colors['C-LSTM'], bold_colors['STA-LSTM']]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    
    # Annotate each bar with the scaled value for clarity
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2e}', 
                ha='center', va='bottom', fontsize=12, color='black')
    
    ax.set_title(f'{label}', fontsize=16)
    ax.set_ylabel('Scaled SD (Relative to Actual)', fontsize=14)

# Set x-axis label for the last subplot
axs[-1].set_xlabel('Method', fontsize=14)

handles = [plt.Rectangle((0,0),1,1, color=color) for color in bold_colors.values()]
labels = [
    'PINN (scaled 1 × 60)', 
    'C-LSTM (scaled 6 × 10¹¹)', 
    'STA-LSTM (scaled 6 × 10¹¹)'
]

# Add the legend to the figure
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.52, 0.93), fontsize=12, title='Legend')
legend.get_title().set_fontsize(14)
legend.get_title().set_weight('bold')
frame = legend.get_frame()
frame.set_facecolor("lightgray")
frame.set_edgecolor("black")    

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('std_plot.png', bbox_inches='tight')
plt.show()
