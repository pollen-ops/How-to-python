import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Settings ---
WINDOW = 10       # Moving average window for derivative
PERCENTILE = 98   # How rare the inflections are (tune as needed)
MIN_BUFFER = 100  # Minimum buffer size in samples (0.2s at 500Hz)
BUFFER_FACTOR = 0.2  # 20% on each side

# --- Load Data ---
data_file = 'test_data_1000rows.txt'
df = pd.read_csv(data_file, sep='\t')
df.columns = df.columns.str.strip()

# --- Calculate Smoothed Derivative ---
df['dPT2_ma'] = df['PT2'].diff().abs().rolling(WINDOW, center=True).mean()
df['dPT3_ma'] = df['PT3'].diff().abs().rolling(WINDOW, center=True).mean()

# --- Auto-threshold ---
thresh_PT2 = np.percentile(df['dPT2_ma'].dropna(), PERCENTILE)
thresh_PT3 = np.percentile(df['dPT3_ma'].dropna(), PERCENTILE)
THRESHOLD = max(thresh_PT2, thresh_PT3)
print(f"Auto-selected threshold: {THRESHOLD:.4f} (percentile={PERCENTILE})")

# --- Detect Inflection Points ---
inflection_mask = ((df['dPT2_ma'] > THRESHOLD) | (df['dPT3_ma'] > THRESHOLD))
inflection_idxs = np.where(inflection_mask)[0]

# --- Define Ranges with Buffer ---
ranges = []
for idx in inflection_idxs:
    buffer = max(int(BUFFER_FACTOR * idx), MIN_BUFFER)
    start = max(idx - buffer, 0)
    end = min(idx + buffer, len(df) - 1)
    if not ranges or start > ranges[-1][1]:
        ranges.append([start, end])
    else:
        ranges[-1][1] = max(ranges[-1][1], end)

# --- Write Each Range of Interest to a CSV ---
base, ext = os.path.splitext(data_file)
output_names = []
for i, (start, end) in enumerate(ranges, 1):
    sub = df.iloc[start:end + 1].copy()
    outname = f"{base}_inflection_{i}.csv"
    sub.to_csv(outname, index=False)
    print(f"Wrote: {outname} (rows {start} to {end})")
    output_names.append((outname, start, end))

# --- Plotting ---
fig, axs = plt.subplots(len(output_names) + 1, 1, figsize=(12, 4 + 2 * len(output_names)), sharex=False)
if not isinstance(axs, np.ndarray): axs = [axs]  # Ensure axs is always iterable

# -- Main plot: full data with inflection regions shaded --
axs[0].plot(df['Time'], df['PT2'], label='PT2', color='blue')
axs[0].plot(df['Time'], df['PT3'], label='PT3', color='red')
for (name, start, end) in output_names:
    axs[0].axvspan(df.loc[start, 'Time'], df.loc[end, 'Time'], color='yellow', alpha=0.3)
axs[0].set_ylabel('Pressure')
axs[0].set_title('Full Dataset with Inflection Regions Highlighted')
axs[0].legend()

# -- Plots for each detected region --
for i, (name, start, end) in enumerate(output_names, 1):
    sub = df.iloc[start:end + 1]
    axs[i].plot(sub['Time'], sub['PT2'], label='PT2', color='blue')
    axs[i].plot(sub['Time'], sub['PT3'], label='PT3', color='red')
    axs[i].set_ylabel('Pressure')
    axs[i].set_title(f'Inflection Region {i}: {name}')
    axs[i].legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
