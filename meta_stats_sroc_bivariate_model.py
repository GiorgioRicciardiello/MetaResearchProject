#!/usr/bin/env python
"""
Extended Python script for comparing diagnostic performance (OSA) at different AHI thresholds.
Each modality corresponds to one AHI threshold (5, 15, and 30).
The input CSV (df_studies.csv) must have the following columns (among others):
    - 'study_id'
    - 'test_data'
    - 'Sensitivity_AHI_5', 'Specificity_AHI_5'
    - 'Sensitivity_AHI_15', 'Specificity_AHI_15'
    - 'Sensitivity_AHI_30', 'Specificity_AHI_30'

For each modality, the script performs:
    1. sROC Analysis (Moses–Littenberg approach) and generates a Table1 summary and sROC curve.
    2. Bivariate Random‐Effects Meta‐Analysis generating a Table2 summary and an ROC plot with a 95% confidence ellipse.
All curves/points for the three modalities are overlaid and labeled for direct comparison.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import logit, expit
from scipy.stats import chi2
from typing import Optional
from config.config import config


# ---------------------------
# sROC Analysis Functions for a modality
# ---------------------------
def sroc_analysis_for_modality(df: pd.DataFrame, sens_col: str, spec_col: str):
    """
    For a given modality, perform the sROC analysis.
    Drops studies with missing sensitivity or specificity for that modality.
    Returns:
      - table1: DataFrame with alpha, beta, DOR at mean S, and Q-point DOR.
      - sroc_curve: tuple (FPR_pred, TPR_pred) for plotting.
      - df_mod: DataFrame with computed values (including TPR, FPR, D, S).
    """
    # Make a copy and drop rows where sensitivity or specificity is missing.
    df_mod = df.copy().dropna(subset=[sens_col, spec_col])
    # Convert sensitivity and specificity to proportions.
    df_mod['TPR'] = df_mod[sens_col] / 100.0
    df_mod['FPR'] = 1 - (df_mod[spec_col] / 100.0)

    # Apply a small continuity correction to avoid issues with logit(0) or logit(1).
    eps = 1e-5
    df_mod['TPR'] = df_mod['TPR'].clip(eps, 1 - eps)
    df_mod['FPR'] = df_mod['FPR'].clip(eps, 1 - eps)

    # Compute D and S.
    df_mod['D'] = logit(df_mod['TPR']) - logit(df_mod['FPR'])
    df_mod['S'] = logit(df_mod['TPR']) + logit(df_mod['FPR'])

    # Fit linear regression: D = alpha + beta * S.
    X = sm.add_constant(df_mod['S'])
    model = sm.OLS(df_mod['D'], X).fit()
    alpha, beta = model.params.const, model.params.S

    # Compute diagnostic odds ratios.
    mean_S = df_mod['S'].mean()
    dor_mean = np.exp(alpha + beta * mean_S)
    dor_qpoint = np.exp(alpha)  # Q-point corresponds to S=0

    table1 = pd.DataFrame({
        'Modality': [sens_col.split('_')[-1]],
        'Intercept (α)': [alpha],
        'Slope (β)': [beta],
        'DOR at mean S': [dor_mean],
        'Q-point DOR (S=0)': [dor_qpoint]
    })

    # Generate sROC curve points.
    S_range = np.linspace(df_mod['S'].min() - 0.5, df_mod['S'].max() + 0.5, 200)
    D_pred = alpha + beta * S_range
    TPR_pred = expit((S_range + D_pred) / 2)
    FPR_pred = expit((S_range - D_pred) / 2)

    return table1, (FPR_pred, TPR_pred), df_mod

def sroc_analysis_for_modality_CI(df: pd.DataFrame, sens_col: str, spec_col: str):
    """
    For a given modality, perform the sROC analysis (Moses–Littenberg).
    Drops studies with missing sensitivity or specificity for that modality.

    Returns:
      - table1: DataFrame with alpha, beta, DOR at mean S, Q-point DOR,
                and their SEs + 95% confidence intervals.
      - sroc_curve: tuple (FPR_pred, TPR_pred) for plotting the sROC curve.
      - df_mod: DataFrame with computed values (including TPR, FPR, D, S).
    """
    # Make a copy and drop rows where sensitivity or specificity is missing.
    df_mod = df.copy().dropna(subset=[sens_col, spec_col])
    # Convert sensitivity and specificity to proportions.
    df_mod['TPR'] = df_mod[sens_col] / 100.0
    df_mod['FPR'] = 1 - (df_mod[spec_col] / 100.0)

    # Apply a small continuity correction to avoid issues with logit(0) or logit(1).
    eps = 1e-5
    df_mod['TPR'] = df_mod['TPR'].clip(eps, 1 - eps)
    df_mod['FPR'] = df_mod['FPR'].clip(eps, 1 - eps)

    # Compute D and S.
    df_mod['D'] = logit(df_mod['TPR']) - logit(df_mod['FPR'])
    df_mod['S'] = logit(df_mod['TPR']) + logit(df_mod['FPR'])

    # Fit linear regression: D = alpha + beta * S.
    X = sm.add_constant(df_mod['S'])
    model = sm.OLS(df_mod['D'], X).fit()

    # Extract parameters and their covariance.
    alpha, beta = model.params.const, model.params.S
    se_alpha, se_beta = model.bse.const, model.bse.S
    cov_matrix = model.cov_params()

    # Compute confidence intervals for alpha and beta.
    # 95% CI = estimate ± 1.96 * SE (assuming normal approximation)
    ci_alpha_lower = alpha - 1.96 * se_alpha
    ci_alpha_upper = alpha + 1.96 * se_alpha
    ci_beta_lower = beta - 1.96 * se_beta
    ci_beta_upper = beta + 1.96 * se_beta

    # Calculate DOR at mean S and Q-point DOR (S=0).
    mean_S = df_mod['S'].mean()
    dor_mean = np.exp(alpha + beta * mean_S)
    dor_qpoint = np.exp(alpha)  # Q-point corresponds to S=0

    # --- Confidence Intervals for DORs ---
    # 1) Q-point DOR = exp(alpha)
    #    Var(log(DOR)) = Var(alpha) = se_alpha^2
    #    => 95% CI for Q-point DOR = exp(alpha ± 1.96*se_alpha)
    qpoint_dor_lower = np.exp(ci_alpha_lower)
    qpoint_dor_upper = np.exp(ci_alpha_upper)

    # 2) DOR at mean S = exp(alpha + beta*mean_S)
    #    Let Z = alpha + beta*mean_S.
    #    Var(Z) = Var(alpha) + mean_S^2 * Var(beta) + 2*mean_S*Cov(alpha,beta).
    var_alpha = cov_matrix.loc["const", "const"]
    var_beta = cov_matrix.loc["S", "S"]
    cov_ab = cov_matrix.loc["const", "S"]
    var_Z = var_alpha + (mean_S**2) * var_beta + 2 * mean_S * cov_ab
    se_Z = np.sqrt(var_Z)

    # 95% CI on log scale => Z ± 1.96*se_Z, then exponentiate.
    Z = alpha + beta * mean_S
    ci_Z_lower = Z - 1.96 * se_Z
    ci_Z_upper = Z + 1.96 * se_Z
    dor_mean_lower = np.exp(ci_Z_lower)
    dor_mean_upper = np.exp(ci_Z_upper)

    # Prepare a DataFrame summarizing the results.
    table1 = pd.DataFrame({
        'Modality': [sens_col.split('_')[-1]],
        'Intercept (α)': [alpha],
        'SE(α)': [se_alpha],
        '95% CI α': [(ci_alpha_lower, ci_alpha_upper)],
        'Slope (β)': [beta],
        'SE(β)': [se_beta],
        '95% CI β': [(ci_beta_lower, ci_beta_upper)],
        'DOR at mean S': [dor_mean],
        '95% CI DOR (mean S)': [(dor_mean_lower, dor_mean_upper)],
        'Q-point DOR (S=0)': [dor_qpoint],
        '95% CI Q-point DOR': [(qpoint_dor_lower, qpoint_dor_upper)]
    })

    # Generate sROC curve points for plotting.
    S_range = np.linspace(df_mod['S'].min() - 0.5, df_mod['S'].max() + 0.5, 200)
    D_pred = alpha + beta * S_range
    TPR_pred = expit((S_range + D_pred) / 2)
    FPR_pred = expit((S_range - D_pred) / 2)

    return table1, (FPR_pred, TPR_pred), df_mod

def plot_all_sroc_curves(sroc_results: dict, output_path: Optional[pathlib.Path] = None):
    """
    Overlays the sROC curves for all modalities.
    sroc_results: dict with key=modality label, value=(sroc_curve, df_mod)
    """
    plt.figure(figsize=(8, 6))
    colors = {'5': 'green', '15': 'orange', '30': 'purple'}

    # Plot each modality's sROC curve and study points.
    for mod, (curve, df_mod) in sroc_results.items():
        FPR_pred, TPR_pred = curve
        plt.plot(FPR_pred, TPR_pred, color=colors[mod], lw=2, label=f'sROC AHI {mod}')
        # Plot individual study points with study_id labels.
        plt.scatter(df_mod['FPR'], df_mod['TPR'], color=colors[mod], alpha=0.6)
        for _, row in df_mod.iterrows():
            plt.annotate(row['study_id'], (row['FPR'], row['TPR']), textcoords="offset points", xytext=(5, 5),
                         fontsize=8)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 2: sROC Curves by AHI Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "Figure2_sroc_all.png")
        print(f"Figure 2 saved to {output_path / 'Figure2_sroc_all.png'}")
    plt.show()


# ---------------------------
# Bivariate Analysis Functions for a modality
# ---------------------------
def bivariate_analysis_for_modality(df: pd.DataFrame, sens_col: str, spec_col: str):
    """
    Perform the bivariate random-effects meta-analysis for one modality.
    Drops studies with missing sensitivity or specificity for that modality.
    Returns:
      - model: Fitted mixed effects model.
      - table2: DataFrame with summary sensitivity, specificity, and DOR.
      - heterogeneity: Dict with between-study variance and correlation.
      - df_long: The long-format DataFrame used for modeling.
      - df_plot: Original study-level DataFrame with sensitivity/spec in probability scale.
    """
    # Make a copy and drop studies missing modality data.
    df_mod = df.copy().dropna(subset=[sens_col, spec_col])
    # Convert percentages to proportions.
    df_mod['sensitivity'] = df_mod[sens_col] / 100.0
    df_mod['specificity'] = df_mod[spec_col] / 100.0

    # Apply continuity corrections.
    eps = 1e-5
    df_mod['sensitivity'] = df_mod['sensitivity'].clip(eps, 1 - eps)
    df_mod['specificity'] = df_mod['specificity'].clip(eps, 1 - eps)

    # Compute logit transforms.
    df_mod['logit_sens'] = logit(df_mod['sensitivity'])
    df_mod['logit_spec'] = logit(df_mod['specificity'])

    # Prepare long format: two rows per study.
    df_long = pd.DataFrame({
        'study_id': np.repeat(df_mod['study_id'].values, 2),
        'measure': np.tile(['sensitivity', 'specificity'], len(df_mod)),
        'logit_value': np.concatenate([df_mod['logit_sens'].values, df_mod['logit_spec'].values])
    })
    df_long['measure_num'] = (df_long['measure'] == 'specificity').astype(int)
    df_long['test_data'] = df_long['study_id'].map(df_mod.set_index('study_id')['test_data'])
    # Replicate rows by sample size (simple weighting)
    df_long = df_long.loc[df_long.index.repeat(df_long['test_data'])].reset_index(drop=True)

    # Fit mixed effects model.
    model = smf.mixedlm("logit_value ~ measure_num", df_long, groups=df_long["study_id"],
                        re_formula="~measure_num").fit()

    beta0 = model.params["Intercept"]
    beta1 = model.params["measure_num"]
    spec_est = beta0 + beta1

    cov_fixed = model.cov_params()
    se_intercept = np.sqrt(cov_fixed.loc["Intercept", "Intercept"])
    var_spec = (cov_fixed.loc["Intercept", "Intercept"] +
                cov_fixed.loc["measure_num", "measure_num"] +
                2 * cov_fixed.loc["Intercept", "measure_num"])
    se_spec = np.sqrt(var_spec)

    ci_sens_logit = (beta0 - 1.96 * se_intercept, beta0 + 1.96 * se_intercept)
    ci_spec_logit = (spec_est - 1.96 * se_spec, spec_est + 1.96 * se_spec)

    summary_sens = expit(beta0)
    summary_spec = expit(spec_est)
    summary_dor = np.exp(beta0 + spec_est)

    table2 = pd.DataFrame({
        'Modality': [sens_col.split('_')[-1]],
        'Sensitivity': [summary_sens],
        'Specificity': [summary_spec],
        'DOR': [summary_dor],
        'CI Sensitivity': [(np.round(expit(ci_sens_logit[0]), 4), np.round(expit(ci_sens_logit[1]), 4))],
        'CI Specificity': [(np.round(expit(ci_spec_logit[0]), 4), np.round(expit(ci_spec_logit[1]), 4))]
    })

    # Heterogeneity from random effects covariance.
    re_cov = model.cov_re
    var_sens_re = re_cov.iloc[0, 0]
    var_spec_re = re_cov.iloc[0, 0] + re_cov.iloc[1, 1] + 2 * re_cov.iloc[0, 1]
    corr = re_cov.iloc[0, 1] / np.sqrt(re_cov.iloc[0, 0] * re_cov.iloc[1, 1])
    heterogeneity = {
        "var_logit_sensitivity": var_sens_re,
        "var_logit_specificity": var_spec_re,
        "correlation": corr
    }

    # For plotting ROC, add FPR = 1 - specificity.
    df_plot = df_mod.copy()
    df_plot['FPR'] = 1 - df_plot['specificity']

    return model, table2, heterogeneity, df_long, df_plot


def plot_bivariate_all(df_plots: dict, model_dict: dict, output_path: Optional[pathlib.Path] = None):
    """
    Overlay individual study ROC plots for all modalities (Figure 1) and the
    bivariate summary with 95% confidence ellipse (Figure 3) for each modality.
    df_plots: dict with modality label as key and corresponding df_plot.
    model_dict: dict with modality label as key and corresponding fitted model.
    """
    colors = {'5': 'blue', '15': 'red', '30': 'green'}

    # Figure 1: Overlay individual study ROC points.
    plt.figure(figsize=(8, 6))
    for mod, df_plot in df_plots.items():
        plt.scatter(df_plot['FPR'], df_plot['sensitivity'], color=colors[mod], alpha=0.6, label=f'AHI {mod}')
        for _, row in df_plot.iterrows():
            plt.annotate(row['study_id'], (row['FPR'], row['sensitivity']), textcoords="offset points", xytext=(5, 5),
                         fontsize=8)
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Figure 1: Individual Study ROC Points by AHI Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "Figure1_ROC_individual_all.png")
        print(f"Figure 1 saved to {output_path / 'Figure1_ROC_individual_all.png'}")
    plt.show()

    # Figure 3: Overlay bivariate confidence ellipses.
    plt.figure(figsize=(8, 6))
    for mod, model in model_dict.items():
        cov_fixed = model.cov_params()
        beta0 = model.params["Intercept"]
        beta1 = model.params["measure_num"]
        mean_logit = np.array([beta0, beta0 + beta1])
        cov_mat = np.array([
            [cov_fixed.loc["Intercept", "Intercept"], cov_fixed.loc["Intercept", "measure_num"]],
            [cov_fixed.loc["Intercept", "measure_num"],
             cov_fixed.loc["Intercept", "Intercept"] + cov_fixed.loc["measure_num", "measure_num"] + 2 * cov_fixed.loc[
                 "Intercept", "measure_num"]]
        ])
        chi2_val = chi2.ppf(0.95, df=2)
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = np.array([np.cos(theta), np.sin(theta)])
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        axes_lengths = np.sqrt(eigvals * chi2_val)
        ellipse_logit = (eigvecs @ (np.diag(axes_lengths) @ circle)) + mean_logit.reshape(2, 1)
        sens_ellipse = expit(ellipse_logit[0, :])
        spec_ellipse = expit(ellipse_logit[1, :])
        x_ellipse = 1 - spec_ellipse
        y_ellipse = sens_ellipse

        plt.plot(x_ellipse, y_ellipse, color=colors[mod], lw=2, label=f'95% CI AHI {mod}')
        summary_sens = expit(beta0)
        summary_spec = expit(beta0 + beta1)
        plt.plot(1 - summary_spec, summary_sens, 'o', color=colors[mod], markersize=8)

    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Figure 3: Bivariate Summary with 95% Confidence Ellipses')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / "Figure3_Bivariate_Ellipse_all.png")
        print(f"Figure 3 saved to {output_path / 'Figure3_Bivariate_Ellipse_all.png'}")
    plt.show()


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Read the studies file.
    try:
        df_studies = pd.read_excel(config.get('papers_manually_selected'))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit(1)

    # Define output folder.
    output_dir = config.get('res_path').joinpath(r'modalities')
    output_dir.mkdir(exist_ok=True)

    # Define modalities and corresponding column names.
    modalities = {
        '5': {'sens': 'Sensitivity_AHI_5', 'spec': 'Specificity_AHI_5'},
        '15': {'sens': 'Sensitivity_AHI_15', 'spec': 'Specificity_AHI_15'},
        '30': {'sens': 'Sensitivity_AHI_30', 'spec': 'Specificity_AHI_30'}
    }

    # Dictionaries to store results.
    sroc_tables = []
    sroc_results = {}  # key: modality label, value: (sroc_curve, df_mod)
    bivar_tables = []
    heterogeneity_dict = {}
    model_dict = {}
    df_plots = {}

    # Loop over modalities.
    for mod, cols in modalities.items():
        # sROC analysis.
        table1, sroc_curve, df_mod = sroc_analysis_for_modality(df_studies, cols['sens'], cols['spec'])
        table1['Modality'] = f"AHI {mod}"
        sroc_tables.append(table1)
        sroc_results[mod] = (sroc_curve, df_mod)

        # Bivariate analysis.
        model, table2, heterogeneity, df_long, df_plot = bivariate_analysis_for_modality(df_studies, cols['sens'],
                                                                                         cols['spec'])
        table2['Modality'] = f"AHI {mod}"
        bivar_tables.append(table2)
        heterogeneity_dict[mod] = heterogeneity
        model_dict[mod] = model
        df_plots[mod] = df_plot

    # Combine and save Table 1 and Table 2.
    combined_table1 = pd.concat(sroc_tables, ignore_index=True)
    combined_table2 = pd.concat(bivar_tables, ignore_index=True)
    print("Combined Table 1 (sROC Analysis):")
    print(combined_table1.to_string(index=False))
    combined_table1.to_csv(output_dir / "Table1_sroc_all.csv", index=False)

    print("\nCombined Table 2 (Bivariate Analysis):")
    print(combined_table2.to_string(index=False))
    combined_table2.to_csv(output_dir / "Table2_bivariate_all.csv", index=False)

    # Plot combined sROC curves (Figure 2).
    plot_all_sroc_curves(sroc_results, output_path=output_dir)

    # Plot individual study ROC points and bivariate summaries.
    plot_bivariate_all(df_plots, model_dict, output_path=output_dir)

    print("\nAll tables and figures have been generated and saved to the 'output' folder.")


