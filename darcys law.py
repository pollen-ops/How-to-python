#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CoolProp.CoolProp import PropsSI

### Setup Constants *** English units ***###
fluid = 'Water'
sample_depth = 0.25 #inches
porous_diameter = 0.5 #inches
venturi_diameter_1 = 0.25 #inches
venturi_diameter_2 = 0.125 #inches


#Loading Raws
data_file = 'test_data_1000rows.txt'
df = pd.read_csv(data_file, sep='\t')
df.columns = df.columns.str.strip()  # Clean up column names

#Quick Data Sanity Check
print(df.head())
print(df.info())

#Unit Conversions
pressure_cols = ['PT1', 'PT2', 'PT3', 'PR12']
for col in pressure_cols:
    df[col + '_Pa'] = df[col] * 6894.76

# Convert temperature columns from F to K (for CoolProp)
temp_cols = ['TC1', 'TC2', 'TC3']
for col in temp_cols:
    df[col + '_K'] = (df[col] - 32) * 5/9 + 273.15

# # Calculate Fluid Properties with CoolProp reference
# df['mu_air'] = df.apply(
#     lambda row: PropsSI('V', 'T', row['TC2_K'], 'P', row['PT2_Pa'], 'Air'),
#     axis=1
#)
def inch_to_meter(length):
    out =  0.0254*length
    return out

def venturi_calculation (row):
    # to find flowrate using venturi : Q =CA_2*sqrt((2*(P_1-P_2)/(rho*(1-(A_2/A_1)^2))
    # C = constant
    # A = constant
    # P = get from data
    # rho = constant/refprop

    #first the constants currently just example values will need to adjust ***
    C = 0.99
    A_1 = np.pi/4*(inch_to_meter(venturi_diameter_1))**2
    A_2 = np.pi/4*(inch_to_meter(venturi_diameter_2))**2
    rho = 1000 #(we will assume incompressible water)

    #Emperical values
    P_1 = row['PT1_Pa']
    P_2 = row['PT2_Pa']

    #Calculation
    Q = C * A_2 * ((2*(P_1 - P_2))/(rho *((A_2/A_1)**2)))**(1/2)

    return Q

def permeability_calculation (row):

    T_K = row['TC2_K']
    P_Pa = row['PT2_Pa']
    Q= venturi_calculation(row)
    mu = PropsSI('V', 'T', T_K, 'P', P_Pa, fluid)
    L = inch_to_meter(sample_depth)
    A = np.pi/4*(inch_to_meter(porous_diameter))**2
    del_P = row['PT2_Pa']-row['PT3_Pa']
    k = (Q*mu*L)/(A*del_P)
    return k


# Calculate flow rate and add as column
df['Q_m3s'] = df.apply(venturi_calculation, axis=1)


df['permeability'] = df.apply(permeability_calculation, axis=1)

print (df.head())





#Visualization
plt.figure(figsize=(10,6))
sns.lineplot(x='Time', y='permeability', data=df)
plt.xlabel('Time (s)')
plt.ylabel('permeability')
plt.title('Pressure Drop vs. Time')
plt.show()
df.to_csv('processed_data.csv', index=False)


#Visualization
plt.figure(figsize=(10,6))
sns.lineplot(x='Time', y='Q_m3s', data=df)
plt.xlabel('Time (s)')
plt.ylabel('flow rate q')
plt.title('Pressure Drop vs. Time')
plt.show()

df.to_csv('processed_data.csv', index=False)

sns.pairplot(df,
             vars=['PT1_Pa', 'PT2_Pa', 'PT3_Pa', 'PR12_Pa', 'TC1_K', 'TC2_K','TC3_K'],
            )
plt.show()


