import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- 1. Streamlit App Setup ---
st.set_page_config(
    page_title="Time Series Anomaly Detection Demo",
    layout="wide"
)

st.title("📊 Time Series Anomaly Detection (Z-Score)")

st.markdown("""
This demo generates a synthetic time series and uses the **Z-Score** method to automatically detect and visualize outlier moments.
""")

# --- 2. Sidebar for User Controls ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    # Anomaly Threshold
    threshold = st.slider(
        "Z-Score Anomaly Threshold ($\sigma$):",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        format="%.1f",
        help="Data points with a Z-Score magnitude greater than this value are flagged as anomalies. Common value is 3.0."
    )

    # Seed for reproducibility
    seed = st.number_input("Random Seed:", value=42, step=1, help="Change the seed to generate a different time series.")
    np.random.seed(seed)
    
    st.markdown("---")
    st.subheader("💡 Anomaly Types")
    st.markdown("""
    * **Point Anomaly:** Single, extreme deviation.
    * **Collective Anomaly:** A small, unusual sequence of points.
    """)

# --- 3. Data Generation (Consistent Logic) ---

# Create 100 days of data
dates = pd.date_range(start='2024-01-01', periods=100)
# Create 'normal' data: random values centered around 50 with a small standard deviation
data = np.random.normal(loc=50, scale=3, size=100)

# Inject Anomaly/Outlier Moments (Points)
data[20] = 75  # Extreme positive spike (Point Anomaly)
data[75] = 40  # Extreme negative dip
data[80:83] = [68, 70, 69] # Collective Anomaly (short sequence of unusual points)

df = pd.DataFrame({'Date': dates, 'Value': data})

# --- 4. Anomaly Detection Logic ---

# Calculate Z-Scores
df['Z_Score'] = zscore(df['Value'])

# Flag Anomalies based on the Streamlit threshold
df['Anomaly'] = (df['Z_Score'].abs() > threshold)

# Isolate the anomalous points for visualization and display
anomalies = df[df['Anomaly']]

# --- 5. Display Results in Streamlit ---

# Create columns for metric and explanation
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detection Metrics")
    st.metric(
        label="Detected Anomalies",
        value=f"{len(anomalies)} points",
        delta=f"Threshold set at {threshold} Standard Deviations"
    )
    if len(anomalies) == 0:
        st.info("No anomalies detected at the current threshold. Try lowering the Z-Score threshold to find subtle outliers.")

with col2:
    st.subheader("Anomalous Moments")
    # Display the table of detected anomalies
    if len(anomalies) > 0:
        st.dataframe(anomalies[['Date', 'Value', 'Z_Score']], use_container_width=True)
    else:
        st.markdown("---")


st.subheader("Time Series Visualization")

# --- 6. Matplotlib Plotting ---

# Create the figure and axes objects for st.pyplot()
fig, ax = plt.subplots(figsize=(12, 5))

# Plot the normal time series data
ax.plot(df['Date'], df['Value'], label='Time Series Value', color='blue', alpha=0.7)

# Highlight the detected outlier moments
ax.scatter(
    anomalies['Date'],
    anomalies['Value'],
    color='red',
    label=f'Anomaly (Z-Score > {threshold})',
    marker='o',
    s=150, 
    zorder=5 
)

# Plot Customization
ax.set_title('Time Series Anomaly Detection (Outlier Moments Highlighted)')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.legend()

# --- Display the Plot in Streamlit (Crucial Step!) ---
# This line avoids the plt.show() pop-up and displays the plot directly in the browser.
st.pyplot(fig)

st.markdown("""


[Image of Z-Score distribution chart]

The Z-Score method is best suited for detecting **point anomalies** that deviate significantly from the mean, assuming the data follows a normal distribution.
""")