import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

# Data
data = {
    'Model': ['PINN', 'LSTM', 'C-LSTM', 'STA-LSTM', 'CT-LSTM'],
    '50x50': [0.9276, 0.9361, 0.8886, 0.9453, 0.9539],
    '100x100': [0.8949, 0.9199, 0.5528, 0.9682, 0.8186],
    '250x250': [0.9168, 0.9306, 0.4738, 0.9357, 0.7768],
}

# Define a dictionary with the bold colors
bold_colors = {
    'PINN': '#414487',
    'LSTM': '#FDEF76',
    'C-LSTM': '#2A788E',
    'STA-LSTM': '#7AD151',
    'CT-LSTM': '#22A884'
}

marker_styles = {
    'PINN': 'x',
    'LSTM': '^',
    'C-LSTM': 's',
    'STA-LSTM': 'D',
    'CT-LSTM': 'p'
}

# Grid dimensions for x-axis
grid_dims = ['50x50', '100x100', '250x250']

fig, ax = plt.subplots(figsize=(8, 6))

# Plot each model's performance across grid dimensions
for model in data['Model']:
    ax.plot(
        grid_dims, 
        [data[dim][data['Model'].index(model)] for dim in grid_dims], 
        label=model, 
        marker=marker_styles[model], 
        color=bold_colors[model], 
        linewidth=3,
        markersize=8
    )

# Set labels and title
ax.set_xlabel('Grid Dimensions', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Accuracy vs Grid Dimensions', fontsize=14)

# Customize y-axis ticks
ax.set_yticks([0.5, 0.6, 0.9, 1.0])

# Disable grid lines
ax.grid(False)

# Add legend
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Legend', borderaxespad=0.)
legend.get_title().set_fontsize('12')
legend.get_title().set_weight('bold')
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_edgecolor("black")

# Make the layout tight
plt.tight_layout()
plt.savefig('correlation_plot_with_custom_shapes.png', bbox_inches='tight')
plt.show()
