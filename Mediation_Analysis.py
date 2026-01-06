#IMPORTS
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#DATA
data = {
    'Province': ['NF', 'NF', 'NF', 'NF', 'NF', 'PE', 'PE', 'PE', 'PE', 'PE', 'NS', 'NS', 'NS', 'NS', 'NS', 'NB', 'NB', 'NB', 'NB', 'NB', 'QC', 'QC', 'QC', 'QC', 'QC', 'ON', 'ON', 'ON', 'ON', 'ON', 'MB', 'MB', 'MB', 'MB', 'MB', 'SK', 'SK', 'SK', 'SK', 'SK', 'AB', 'AB', 'AB', 'AB', 'AB', 'BC', 'BC', 'BC', 'BC', 'BC'],
    'Time_frame': ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024'],
    'Year_Start': [2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023, 2019, 2020, 2021, 2022, 2023],
    'Population': [520100, 521100, 522600, 523500, 534600, 158400, 162200, 164700, 170800, 176600, 971200, 979600, 993400, 1018600, 1046300, 776800, 781500, 788100, 808900, 832700, 8537700, 8605500, 8648500, 8733900, 8848400, 14595300, 14755200, 14951700, 15286900, 15729100, 1373900, 1381500, 1391100, 1409100, 1438300, 1174500, 1182200, 1187800, 1205100, 1228000, 4371100, 4413400, 4441700, 4554400, 4697500, 5071300, 5119500, 5160900, 5319400, 5477500], # July 1st population estimate
    'Hab_yr': [316, 4178, 324, 23886, 21913, 15.3, 1388, 0.1, 0, 8, 154.5, 709, 197, 3389, 25093, 227.7, 1388, 427, 176, 854, 9604, 59748, 49985, 29638, 439888, 269631.5, 15480, 793326, 2561, 441474, 6415.3, 49527, 1266777, 165078, 189782, 47737.7, 42160, 956084, 244275, 1850829, 883414, 2275, 54060, 137310, 1951299, 20986, 14446, 869255, 135032, 2840754], # Hectares Burned (X)
    'Days_yr': [78682, 68948, 74328, 80253, 76995, 35846, 24770, 29935, 25272, 27793, 167425, 186610, 161328, 182863, 196269, 178736, 112920, 167402, 176083, 205820, 1431206, 1243768, 1276876, 1386087, 1434667, 2176841, 1968996, 2196992, 2183266, 2056881, 269715, 235708, 239162, 203383, 209606, 124487, 178988, 142254, 175702, 177031, 781700, 698734, 680499, 752646, 738241, 663013, 596338, 647149, 682471, 726514], # Days Stayed (M)
    'OD_yr': [17, 25, 25, 29, 37, 5, 8, 12, 6, 9, 56, 49, 40, 62, 73, 35, 35, 45, 76, 98, 213, 322, 296, 315, 405, 1605, 2518, 2961, 2531, 2639, 62, 233, 286, 291, 382, 115, 250, 327, 297, 347, 1030, 1801, 2330, 2417, 2624, 262, 1192, 1623, 1523, 1867] # Opioid Deaths (Y)
}
df = pd.DataFrame(data)

#CALCULATIONS
# Convert all variables to numeric and sort for proper lagging
df['Hab_yr'] = pd.to_numeric(df['Hab_yr'], errors='coerce')
df['Days_yr'] = pd.to_numeric(df['Days_yr'], errors='coerce')
df['OD_yr'] = pd.to_numeric(df['OD_yr'], errors='coerce')
df['Year_Start'] = pd.to_numeric(df['Year_Start'], errors='coerce')
df = df.dropna().sort_values(by=['Province', 'Year_Start'])

# Calculation of Per Capita Rates (M and Y) and Lagging (X) 
# Scale factor of 100,000 for rate calculation
SCALE = 100000 
df['M_rate'] = (df['Days_yr'] / df['Population']) * SCALE # M: Hospital Days per 100k
df['Y_rate'] = (df['OD_yr'] / df['Population']) * SCALE   # Y: Opioid Deaths per 100k
# Create Lagged Variable (X_t-1)
df['Hab_yr_Lag1'] = df.groupby('Province')['Hab_yr'].shift(1)
df_lagged = df.dropna(subset=['Hab_yr_Lag1']).copy()

#Final Z-Score Standardization for Regression
# X: Hab_yr_Lag1 (Previous year's fires in hectares burned)
# M: M_rate (Current year's days stayed RATE)
# Y: Y_rate (Current year's opioid deaths RATE)
df_lagged['X'] = (df_lagged['Hab_yr_Lag1'] - df_lagged['Hab_yr_Lag1'].mean()) / df_lagged['Hab_yr_Lag1'].std()
df_lagged['M'] = (df_lagged['M_rate'] - df_lagged['M_rate'].mean()) / df_lagged['M_rate'].std()
df_lagged['Y'] = (df_lagged['Y_rate'] - df_lagged['Y_rate'].mean()) / df_lagged['Y_rate'].std()

# Bootstrapping Procedure for Indirect Effect (a*b)
np.random.seed(42) # Set seed for reproducibility
B = 20000  # Number of bootstrap samples
indirect_effects = []
for i in range(B):
    # Resample the data (with replacement)
    sample_df = df_lagged.sample(n=len(df_lagged), replace=True)
    # Run Path A: M ~ X
    try:
        model_a = smf.ols('M ~ X', data=sample_df).fit()
#OLS regression finds the line of best fit for the data through adjusting slope and intercept
#Justified using the Guass-Markov theorem, OLS provides the Best Linear Unbiased Estimators (BLUE) for coefficients in linear regression models
        a = model_a.params['X']
    except Exception:
        continue
    # Run Path B and C': Y ~ X + M
    try:
        model_bc = smf.ols('Y ~ X + M', data=sample_df).fit()
        b = model_bc.params['M']
    except Exception:
        continue
    # Calculate Indirect Effect: a * b
    indirect_effects.append(a * b)

# Calculate the 95% Confidence Interval (CI) from the bootstrap distribution
indirect_effects = np.array(indirect_effects)
lower_bound = np.percentile(indirect_effects, 2.5)
upper_bound = np.percentile(indirect_effects, 97.5)

# Final Non-Bootstrapped Models for Point Estimates 
model_a_final = smf.ols('M ~ X', data=df_lagged).fit()
a_point = model_a_final.params['X']
model_bc_final = smf.ols('Y ~ X + M', data=df_lagged).fit()
b_point = model_bc_final.params['M']
direct_effect_c_prime = model_bc_final.params['X']
model_c_final = smf.ols('Y ~ X', data=df_lagged).fit()
total_effect_c = model_c_final.params['X']

# Calculate the point estimate for the indirect effect
indirect_effect_point = a_point * b_point
print(f"Sample Size: {len(df_lagged)}")
print(f"Indirect Effect (a*b): {indirect_effect_point:.4f}")
print(f"95% CI Lower Bound: {lower_bound:.4f}")
print(f"95% CI Upper Bound: {upper_bound:.4f}")
print(f"Total Effect (c): {total_effect_c:.4f}")
print(f"Direct Effect (c'): {direct_effect_c_prime:.4f}")

#RESULTS:
#Bootstrap samples:20,000
#Indirect Effect (a*b): 0.0409
#-of the estimated effect of .2980, .0409 of that change is specifically explained by X→M→Y
#95% CI Lower Bound: -0.0000
#95% CI Upper Bound: 0.1183
#-CI boundaries are close to zero, but with number of iterations effect is real but likely weak
#Total Effect (c): 0.2980
#-for every 1-unit increase in forest fire rates there is a 0.2980-unit increase in opioid-related deaths
#Direct Effect (c'): 0.2571
#-for every 1-unit increase in forest fire rates there is a .2571-unit increase in opioid-related deaths, after controlling for M
#-the direct effect is 6.3X larger than the indirect effect (.2571/.0409=6.3), 86.3% remains unexplained
#Indirect/total=13.7% (amount of data channeled through pathway)
#-86.3% remains unexplained (ex/economic social factors, health systems, other environmental factors, also hospitalizations only include acute distress)
#-undeniable significance but small magnitude
#Forest fires rates are associated with with up to a 13.7% increase in opioid-related deaths, specifically through a transmitted pathway involving high distress characterized by mental health hospitalizations

