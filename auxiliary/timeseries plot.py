import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import matplotlib.font_manager as fm


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

# Calculate averages and standard deviations
y_test_avg = np.mean(y_test, axis=(1, 2))
y_pred_avg_pinn = np.mean(y_pred_pinn, axis=(1, 2))
y_pred_avg_clstm = np.mean(y_pred_clstm, axis=(1, 2))
y_pred_avg_stalstm = np.mean(y_pred_stalstm, axis=(1, 2))

# Define a dictionary with the bold colors
bold_colors = {
    'PINN': '#414487',
    'C-LSTM': '#2A788E',
    'STA-LSTM': '#7AD151'
}

# Define labels and indices for plotting
labels_to_plot = ['IL-8 (Cytokine)', 'IL-6 (Cytokine)', 'TGF (Cytokine)']
indices_to_plot = [0, 2, 5] 

# Calculate means and standard deviations for display
mean_std_info = {}
for i, label in zip(indices_to_plot, labels_to_plot):
    mean_std_info[label] = {
        'Actual Mean': np.mean(y_test_avg[:, i]),
        'Actual Std': np.std(y_test_avg[:, i]),
        'PINN Mean': np.mean(y_pred_avg_pinn[:, i]),
        'PINN Std': np.std(y_pred_avg_pinn[:, i]),
        'C-LSTM Mean': np.mean(y_pred_avg_clstm[:, i]),
        'C-LSTM Std': np.std(y_pred_avg_clstm[:, i]),
        'STA-LSTM Mean': np.mean(y_pred_avg_stalstm[:, i]),
        'STA-LSTM Std': np.std(y_pred_avg_stalstm[:, i])
    }

# Plotting
output_dir = 'plots-Comparison'
os.makedirs(output_dir, exist_ok=True)

fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)  # Create a figure with 3 subplots stacked vertically

custom_limits = [(1.4e-8, 2.5e-8), (3e-12, 1e-10), (6e-11, 3e-9)]

for ax, i, label, (lower, upper) in zip(axs, indices_to_plot, labels_to_plot, custom_limits):
    # Plot actual data
    ax.plot(time_steps, y_test_avg[:, i], label='Actual', color='lightsteelblue', marker='o', linestyle='-')

    # Plot PINN
    ax.plot(time_steps, y_pred_avg_pinn[:, i], label='PINN', color=bold_colors['PINN'], marker='x', linestyle='-')

    # Plot C-LSTM
    ax.plot(time_steps, y_pred_avg_clstm[:, i], label='C-LSTM', color=bold_colors['C-LSTM'], marker='s', linestyle='-')

    # Plot STA-LSTM
    ax.plot(time_steps, y_pred_avg_stalstm[:, i], label='STA-LSTM', color=bold_colors['STA-LSTM'], marker='d', linestyle='-')

    ax.set_ylabel('Concentration', fontsize=16)
    ax.set_yscale('log')
    ax.set_ylim(lower, upper)
    ax.set_title(label, fontsize=16)

# Function to convert a number to scientific notation with powers of 10 and superscript formatting
def format_power_of_ten(value):
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    coefficient = value / 10**exponent
    # Format with LaTeX for the exponent
    return f"{coefficient:.2f} × 10$^{{{exponent}}}$"

# Construct the info text for mean and standard deviation
mean_std_info_text = "Mean and Standard Deviation\n\n"

# Iterate through each cytokine label and its corresponding stats
for label, stats in mean_std_info.items():
    # Format the info text for each cytokine
    mean_std_info_text += f"{label}\n"
    mean_std_info_text += f"  Actual Mean: {format_power_of_ten(stats['Actual Mean'])}\n Actual Std: {format_power_of_ten(stats['Actual Std'])}\n"
    mean_std_info_text += f"  PINN Mean: {format_power_of_ten(stats['PINN Mean'])}\n PINN Std: {format_power_of_ten(stats['PINN Std'])}\n"
    mean_std_info_text += f"  C-LSTM Mean: {format_power_of_ten(stats['C-LSTM Mean'])}\n C-LSTM Std: {format_power_of_ten(stats['C-LSTM Std'])}\n"
    mean_std_info_text += f"  STA-LSTM Mean: {format_power_of_ten(stats['STA-LSTM Mean'])}\n STA-LSTM Std: {format_power_of_ten(stats['STA-LSTM Std'])}\n"
    mean_std_info_text += "\n"

# Display the mean and standard deviation information in a text box above the legend
fig.text(1.1, 0.90, mean_std_info_text, fontsize=16, verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))


plt.xlabel('Time', fontsize=16)
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.15), ncol=1, fontsize=16, title='Legend')
legend.get_title().set_fontsize('18') 
legend.get_title().set_weight('bold')
frame = legend.get_frame()
frame.set_facecolor("white") 
frame.set_edgecolor("black") 

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend and subplots properly
plt.savefig('timeseries plot.png', bbox_inches='tight')
plt.show()
