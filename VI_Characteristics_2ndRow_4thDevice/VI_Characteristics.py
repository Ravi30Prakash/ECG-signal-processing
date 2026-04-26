import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r"C:\Users\Admin\Downloads\ECG_Received\Ravi 22nd April2026\IV between 5-7.csv"

# === STEP 1: Find correct header row ===
with open(file_path, 'r') as f:
    lines = f.readlines()

header_row = None
for i, line in enumerate(lines):
    line_lower = line.lower()
    if 'time' in line_lower and 'volt' in line_lower and 'curr' in line_lower:
        header_row = i
        break

if header_row is None:
    raise Exception("Could not find header row with Time, Voltage, Current")

# === STEP 2: Read correct data ===
data = pd.read_csv(file_path, skiprows=header_row)

# Clean column names
data.columns = data.columns.str.strip()

print("Columns detected:", data.columns)

# === STEP 3: Extract columns ===
time_col = [c for c in data.columns if 'time' in c.lower()][0]
voltage_col = [c for c in data.columns if 'volt' in c.lower()][0]
current_col = [c for c in data.columns if 'curr' in c.lower()][0]

V = pd.to_numeric(data[voltage_col], errors='coerce')
I = pd.to_numeric(data[current_col], errors='coerce')

mask = ~(V.isna() | I.isna())
V = V[mask]
I = I[mask]

# === STEP 4: Plot I-V ===
plt.figure()
plt.plot(V, I, marker='o')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('I-V Characteristics')
plt.grid()
plt.show()

# === STEP 5: Resistance ===
R = np.where(I != 0, V / I, np.nan)

plt.figure()
plt.plot(V, R, marker='o')
plt.xlabel('Voltage (V)')
plt.ylabel('Resistance (Ohms)')
plt.title('Resistance vs Voltage')
plt.grid()
plt.show()

# === STEP 6: Results ===
print("Average Resistance:", np.nanmean(R))

coeffs = np.polyfit(I, V, 1)
print("Resistance from slope:", coeffs[0])