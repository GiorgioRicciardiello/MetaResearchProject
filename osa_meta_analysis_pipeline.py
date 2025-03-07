import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.meta_analysis import combine_effects
from config.config import config
import seaborn as sns

# ===============================
# 1. Data Ingestion and Cleaning
# ===============================

# Read the Excel file (adjust the path as needed)
df = pd.read_excel(config.get('papers_manually_selected'))
df.drop(columns=['Unnamed: 3'], inplace=True)
print("Original columns:", df.columns.tolist())

# Expected columns include:
# 'Study_ID', 'Sample_Size', 'k_fold', 'Train_Data', 'Test_Data',
# 'Sensitivity_5', 'Specificity_5', 'AUC_5',
# 'Sensitivity_15', 'Specificity_15', 'AUC_15',
# 'Sensitivity_30', 'Specificity_30', 'AUC_30'

# Reshape the data from wide to long format using pd.wide_to_long.
# This will create a new column 'AHI_cutoff' (with values 5, 15, 30)
# and combine the metric columns (Sensitivity, Specificity, AUC) into single columns.
df_long = pd.wide_to_long(
    df,
    stubnames=['Sensitivity', 'Specificity', 'AUC'],
    i=['study_id', 'References', 'study_name', 'sample_size', 'k-fold', 'train_data', 'test_data'],
    j='AHI_cutoff',
    sep='_AHI_',
    suffix='\d+'
).reset_index()

print("Reshaped DataFrame preview:")
print(df_long.head())

# If your metric values are given in percentages, convert them to proportions:
for metric in ['Sensitivity', 'Specificity', 'AUC']:
    df_long[metric] = df_long[metric] / 100.0

# Drop rows that have missing metric values (if any)
df_long = df_long.dropna(subset=['Sensitivity', 'Specificity', 'AUC'])

# ===============================
# 2. Compute Variances for Sensitivity and Specificity
# ===============================
# For sensitivity:
#   variance = (p * (1 - p)) / n_pos
# For specificity:
#   variance = (p * (1 - p)) / n_neg
# Here, we assume "test_data" approximates the total sample,
# and we use half of that value for each (as a proxy).

df_long['n_pos'] = df_long['test_data'].apply(lambda x: x / 2)
df_long['n_neg'] = df_long['test_data'].apply(lambda x: x / 2)
df_long['var_sens'] = (df_long['Sensitivity'] * (1 - df_long['Sensitivity'])) / df_long['n_pos']
df_long['var_spec'] = (df_long['Specificity'] * (1 - df_long['Specificity'])) / df_long['n_neg']

print("Cleaned and reshaped data preview:")
print(df_long.head())

# ===============================
# 3. Exploratory Analysis
# ===============================

# Summary statistics
print("\nSummary statistics:")
print(df_long.describe())

# Plot distributions of Sensitivity and Spec by AHI cutoff:
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
sns.violinplot(x='AHI_cutoff', y='Sensitivity', data=df_long, inner='box', color='.8')
sns.stripplot(x='AHI_cutoff', y='Sensitivity', data=df_long, color='black', alpha=0.6)
plt.xlabel('AHI Cutoff')
plt.ylabel('Sensitivity (proportion)')
plt.title('Distribution of Sensitivity by AHI Cutoff')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x='AHI_cutoff', y='Specificity', data=df_long, inner='box', color='.8')
sns.stripplot(x='AHI_cutoff', y='Specificity', data=df_long, color='black', alpha=0.6)
plt.xlabel('AHI Cutoff')
plt.ylabel('Specificity (proportion)')
plt.title('Distribution of Specificity by AHI Cutoff')
plt.tight_layout()
plt.show()



# ===============================
# 4. Meta-Analysis Modeling: Pooled Sensitivity
# ===============================

# Here we use the inverse-variance method with a random-effects model (DerSimonian-Laird)
# for a simple univariate meta-analysis of sensitivity.
# (Note: For diagnostic test accuracy, ideally a bivariate model would be used to jointly model sensitivity and specificity.)

def run_meta_analysis(subgroup, effect_col, var_col, method_re="dl"):
    """
    Perform a random-effects meta-analysis using statsmodels' combine_effects
    and return the pooled estimate plus 95% confidence interval from the random-effects model.
    """
    effects = subgroup[effect_col].values
    variances = subgroup[var_col].values

    # Compute the meta-analysis using the DerSimonian-Laird method (or other)
    results = combine_effects(effects, variances, method_re=method_re)
    summary_df = results.summary_frame(alpha=0.05, use_t=False)

    # Extract the random-effects row
    random_eff_row = summary_df.loc['random effect']
    pooled_mean = random_eff_row['eff']
    ci_low = random_eff_row['ci_low']
    ci_upp = random_eff_row['ci_upp']

    return pooled_mean, (ci_low, ci_upp)

# Run meta-analysis for each AHI cutoff
print("\nPooled Sensitivity by AHI Cutoff:")
cutoffs = sorted(df_long['AHI_cutoff'].unique())

# ===============================
# 5. Meta-Analysis: Pooled Sensitivity by AHI Cutoff
# ===============================
results = []

# Loop over each cutoff value
for cutoff in cutoffs:
    # Subset the data for the current cutoff
    subgroup = df_long[df_long['AHI_cutoff'] == cutoff]

    # Run meta-analysis for Sensitivity
    pooled_sens, (ci_low_sens, ci_upp_sens) = run_meta_analysis(subgroup, 'Sensitivity', 'var_sens', method_re="dl")

    # Run meta-analysis for Specificity
    pooled_spec, (ci_low_spec, ci_upp_spec) = run_meta_analysis(subgroup, 'Specificity', 'var_spec', method_re="dl")

    # Append the metrics to the results list
    results.append({
        'AHI_cutoff': cutoff,
        'pooled_sensitivity': pooled_sens,
        'ci_low_sens': ci_low_sens,
        'ci_upp_sens': ci_upp_sens,
        'pooled_specificity': pooled_spec,
        'ci_low_spec': ci_low_spec,
        'ci_upp_spec': ci_upp_spec
    })

# Combine the list of dictionaries into a DataFrame
df_pooled_metrics = pd.DataFrame(results)

# Print the DataFrame to check the output
print("Pooled Metrics by AHI Cutoff:")
print(df_pooled_metrics)

# ===============================
# 4. Visualization: Forest Plot for Sensitivity
# ===============================

def forest_plot(effects, variances, study_labels, title):
    se = np.sqrt(variances)
    ci_lower = effects - 1.96 * se
    ci_upper = effects + 1.96 * se
    y_pos = np.arange(len(effects))

    plt.figure(figsize=(8, len(effects) * 0.5 + 2))
    plt.errorbar(effects, y_pos, xerr=1.96 * se, fmt='o', color='black', ecolor='gray', capsize=3)
    plt.yticks(y_pos, study_labels)
    plt.axvline(x=np.mean(effects), color='red', linestyle='--', label='Mean effect')
    plt.xlabel('Sensitivity (proportion)')
    plt.title(title)
    plt.gca().invert_yaxis()  # highest study at the top
    plt.legend()
    plt.show()


# Create a forest plot for one cutoff (e.g., AHI > 15)
cutoff_to_plot = 15
subgroup_plot = df_long[df_long['AHI_cutoff'] == cutoff_to_plot]
forest_plot(subgroup_plot['Sensitivity'], subgroup_plot['var_sens'],
            subgroup_plot['Study_ID'], f'Forest Plot of Sensitivity (AHI > {cutoff_to_plot})')


# ===============================
# 5. Heterogeneity Assessment for Sensitivity
# ===============================

def calculate_I2(effects, variances):
    # Weighted average of effects (fixed-effect weights)
    weights = 1 / variances
    fixed_effect = np.average(effects, weights=weights)
    Q = np.sum(weights * (effects - fixed_effect) ** 2)
    df_Q = len(effects) - 1
    I2 = max(0, (Q - df_Q) / Q) * 100 if Q > df_Q else 0
    return I2


# Heterogeneity assessment for Sensitivity and Specifcity
heterogeneity_results = []

# Loop over each cutoff value
for cutoff in cutoffs:
    # Subset the data for the current cutoff
    subgroup = df_long[df_long['AHI_cutoff'] == cutoff]

    # Calculate I² for Sensitivity
    I2_sens = calculate_I2(subgroup['Sensitivity'].values, subgroup['var_sens'].values)

    # Calculate I² for Specificity
    I2_spec = calculate_I2(subgroup['Specificity'].values, subgroup['var_spec'].values)

    # Append the results to the list
    heterogeneity_results.append({
        'AHI_cutoff': cutoff,
        'I2_sensitivity': I2_sens,
        'I2_specificity': I2_spec
    })

# Combine the list into a DataFrame
df_heterogeneity = pd.DataFrame(heterogeneity_results)

# Print the DataFrame to check the output
print("Heterogeneity Assessment Metrics by AHI Cutoff:")
print(df_heterogeneity)
# ===============================
# 6. Publication Bias: Funnel Plot for Sensitivity
# ===============================

plt.figure(figsize=(8, 6))
precision = 1 / np.sqrt(df_long['var_sens'])
plt.scatter(precision, df_long['Sensitivity'], alpha=0.7)
plt.xlabel('Precision (1/SE)')
plt.ylabel('Sensitivity (proportion)')
plt.title('Funnel Plot for Sensitivity (All Studies)')
plt.show()

# ===============================
# 7.  Further Analysis
#  - Using a bivariate random-effects model (or hierarchical SROC model)
#    to jointly analyze sensitivity and specificity. This can be implemented using
#    Bayesian methods (e.g., PyMC3 or Stan) or by calling R packages via rpy2.
# we have the script meta_stats_sroc_bivariate_model.py
