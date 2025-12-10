import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- 1. Generate Synthetic Time Series Data ---
# Set seed for reproducibility
np.random.seed(42)

# Create 100 days of data
dates = pd.date_range(start='2024-01-01', periods=100)
# Create 'normal' data: random values centered around 50 with a small standard deviation
data = np.random.normal(loc=50, scale=3, size=100)

# Inject Anomaly/Outlier Moments (Points)
data[20] = 75  # Extreme positive spike (Point Anomaly)
data[75] = 40  # Extreme negative dip
data[80:83] = [68, 70, 69] # Collective Anomaly (short sequence of unusual points)

df = pd.DataFrame({'Date': dates, 'Value': data})

# --- 2. Calculate Z-Scores ---
# The Z-Score tells us how many standard deviations a data point is from the mean.
df['Z_Score'] = zscore(df['Value'])

# --- 3. Flag Anomalies based on a Threshold ---
# A common threshold for normal distributions is 3 standard deviations (a Z-Score of |Z| > 3).
threshold = 3.0
df['Anomaly'] = (df['Z_Score'].abs() > threshold)

# Isolate the anomalous points for visualization
anomalies = df[df['Anomaly']]

# --- 4. Visualize the Outlier Moments ---
print(f"--- Detected Anomaly Moments (Z-Score > {threshold}) ---")
print(anomalies[['Date', 'Value', 'Z_Score']])
print("-" * 50)

# Create the plot
plt.figure(figsize=(14, 6))

# Plot the normal time series data
plt.plot(df['Date'], df['Value'], label='Time Series Value', color='blue', alpha=0.7)

# Highlight the detected outlier moments
plt.scatter(
    anomalies['Date'],
    anomalies['Value'],
    color='red',
    label=f'Anomaly (Z-Score > {threshold})',
    marker='o',
    s=100, # size of the marker
    zorder=5 # ensure the scatter points are on top
)

plt.title('Time Series Anomaly Detection using Z-Score')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Display a diagram illustrating the components of a time series
# 

plt.show()