"""
https://www.youtube.com/watch?v=FQ5pkNsi-e0&ab_channel=StellenboschFacultyofMedicine%26HealthSciences

Hierarchal sROC approach
- Rutter - Gtsonis (StatMed 2001)

"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config.config import config
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Dict, List
from scipy.special import expit, logit
import pathlib
from scipy.stats import norm


def compute_Q_and_I2(effects, variances):
    """
    Compute Cochran's Q and I² for a meta-analysis.

    Parameters
    ----------
    effects : array-like
        List or array of effect sizes from each study.
    variances : array-like
        List or array of variances corresponding to each effect size.

    Returns
    -------
    pooled_effect : float
        The pooled effect size using inverse-variance weighting.
    Q : float
        Cochran's Q statistic.
    I2 : float
        I² statistic (percentage of variability due to heterogeneity).
    """
    effects = np.array(effects)
    variances = np.array(variances)
    # Compute inverse-variance weights
    weights = 1 / variances
    # Compute the fixed-effect pooled estimate (weighted average)
    pooled_effect = np.sum(weights * effects) / np.sum(weights)
    # Calculate Cochran's Q statistic: weighted sum of squared differences
    Q = np.sum(weights * (effects - pooled_effect) ** 2)
    # Number of studies
    k = len(effects)
    # Calculate I² statistic. Guard against division by zero.
    I2 = 0 if Q == 0 else max(0, (Q - (k - 1)) / Q) * 100
    return pooled_effect, Q, I2


def hierarchical_summary_roc(df_study: pd.DataFrame):
    """
    Fit a simplified HSROC model using a mixed effects model on the logit-transformed sensitivity and specificity.
    The study's sample size is incorporated by replicating rows according to the 'test_data' column.

    Parameters:
      df (pd.DataFrame): DataFrame with columns 'study_id', 'sensitivity', 'specificity', and 'test_data'

    Returns:
      model: Fitted mixed effects model
      summary_points: Dictionary with summary sensitivity and specificity estimates
    """
    # Copy the input DataFrame
    df = df_study.copy()
    df['specificity'] = df['specificity'] / 100
    df['sensitivity'] = df['sensitivity'] / 100
    # Transform sensitivity and specificity to the logit scale.
    # (If values are exactly 0 or 1, consider applying a small correction.)
    df['logit_sens'] = logit(df['sensitivity'])
    df['logit_spec'] = logit(df['specificity'])

    # Convert to long format: two rows per study (one for sensitivity, one for specificity)
    long_df = pd.DataFrame({
        'study_id': np.repeat(df['study_id'].values, 2),
        'measure': np.tile(['sensitivity', 'specificity'], len(df)),
        'logit_value': np.concatenate([df['logit_sens'].values, df['logit_spec'].values])
    })

    # Create a numeric indicator: 0 for sensitivity, 1 for specificity
    long_df['measure_num'] = (long_df['measure'] == 'specificity').astype(int)

    # Map the sample size from df into the long format
    long_df['test_data'] = long_df['study_id'].map(df.set_index('study_id')['test_data'])

    # Replicate rows according to sample size (each row is repeated test_data times)
    # This is a simple approach to weight studies by their sample size.
    long_df = long_df.loc[long_df.index.repeat(long_df['test_data'])].reset_index(drop=True)

    # Fit a linear mixed effects model with a random intercept and random slope for each study.
    # Fixed effect: measure_num. The intercept estimates logit(sensitivity) and the slope gives the difference.
    model = smf.mixedlm("logit_value ~ measure_num", long_df, groups=long_df["study_id"],
                        re_formula="~measure_num").fit()

    # Extract fixed effects:
    beta0 = model.params["Intercept"]
    beta1 = model.params["measure_num"]

    summary_logit_sens = beta0
    summary_logit_spec = beta0 + beta1

    summary_sens = expit(summary_logit_sens)
    summary_spec = expit(summary_logit_spec)

    summary_points = {
        "sensitivity": summary_sens,
        "specificity": summary_spec
    }

    # Plot the summary point along with individual study points (back-transformed)
    plt.figure(figsize=(8, 6))
    plt.scatter(1 - df['specificity'], df['sensitivity'], c='gray', alpha=0.7, label='Studies')
    plt.scatter(1 - summary_spec, summary_sens, c='red', s=100, label='Summary (HSROC)')

    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Hierarchical Summary ROC (HSROC) Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, summary_points


def bivariate_meta_analysis_with_CI(df_study: pd.DataFrame,
                                    output_path: Optional[pathlib.Path],
                                    title: str = 'ROC Space by Study ID',
                                    file_name: str = 'sroc_space.png'
                                    ):
    """
    Fit a bivariate random-effects meta-analysis model using a single linear mixed model.
    This function estimates between-study variation in logit-sensitivity and logit-specificity
    separately (and their correlation) and provides summary estimates (with 95% confidence intervals)
    for the mean logit sensitivity and specificity. Confidence intervals account for heterogeneity
    beyond chance between studies.

    The summary estimates are provided on three scales:
      - Logit scale (raw fixed-effects estimates)
      - Probability scale (0-1)
      - Percentage units (0-100)

    Parameters:
      df_study (pd.DataFrame): DataFrame with columns:
          - 'study_id'
          - 'sensitivity' (in percentage)
          - 'specificity' (in percentage)
          - 'test_data' (sample size for the study)

    Returns:
      model: Fitted mixed effects model
      summary_points: Dictionary with summary estimates (on logit, probability, and percentage scales)
      heterogeneity: Dictionary with between-study variance estimates and correlation
    """
    # Copy the data and convert percentages to proportions
    df = df_study.copy()
    df['sensitivity'] = df['sensitivity'] / 100
    df['specificity'] = df['specificity'] / 100

    # Compute logit transforms
    # (In practice, a continuity correction may be needed if any values are exactly 0 or 1.)
    df['logit_sens'] = logit(df['sensitivity'])
    df['logit_spec'] = logit(df['specificity'])

    # Convert to long format: two rows per study (one for sensitivity, one for specificity)
    long_df = pd.DataFrame({
        'study_id': np.repeat(df['study_id'].values, 2),
        'measure': np.tile(['sensitivity', 'specificity'], len(df)),
        'logit_value': np.concatenate([df['logit_sens'].values, df['logit_spec'].values])
    })
    # Create a numeric indicator: 0 for sensitivity and 1 for specificity.
    long_df['measure_num'] = (long_df['measure'] == 'specificity').astype(int)
    # Map the sample size from the original df into the long format
    long_df['test_data'] = long_df['study_id'].map(df.set_index('study_id')['test_data'])
    # Replicate rows according to sample size (simple weighting)
    long_df = long_df.loc[long_df.index.repeat(long_df['test_data'])].reset_index(drop=True)

    # Fit the mixed effects model with a random intercept and random slope (by study)
    # Model: logit_value = Intercept + beta1*measure_num + (u0 + u1*measure_num) + error
    model = smf.mixedlm("logit_value ~ measure_num", long_df, groups=long_df["study_id"],
                        re_formula="~measure_num").fit()

    # Extract fixed effects:
    # For sensitivity (measure_num=0): summary logit sensitivity = beta0.
    # For specificity (measure_num=1): summary logit specificity = beta0 + beta1.
    beta0 = model.params["Intercept"]
    beta1 = model.params["measure_num"]
    spec_est = beta0 + beta1

    # Obtain the covariance matrix for fixed effects
    cov_fixed = model.cov_params()
    se_intercept = np.sqrt(cov_fixed.loc["Intercept", "Intercept"])
    # For specificity, we need the SE of (beta0+beta1)
    var_spec = (cov_fixed.loc["Intercept", "Intercept"] +
                cov_fixed.loc["measure_num", "measure_num"] +
                2 * cov_fixed.loc["Intercept", "measure_num"])
    se_spec = np.sqrt(var_spec)

    # Calculate 95% confidence intervals on the logit scale
    ci_sens_logit = (beta0 - 1.96 * se_intercept, beta0 + 1.96 * se_intercept)
    ci_spec_logit = (spec_est - 1.96 * se_spec, spec_est + 1.96 * se_spec)

    # Transform summary estimates and CIs back to the probability scale
    summary_sens = expit(beta0)
    summary_spec = expit(spec_est)
    ci_sens = (expit(ci_sens_logit[0]), expit(ci_sens_logit[1]))
    ci_spec = (expit(ci_spec_logit[0]), expit(ci_spec_logit[1]))

    # Additionally, transform the probabilities into percentage units.
    summary_sens_pct = summary_sens * 100
    summary_spec_pct = summary_spec * 100
    ci_sens_pct = (ci_sens[0] * 100, ci_sens[1] * 100)
    ci_spec_pct = (ci_spec[0] * 100, ci_spec[1] * 100)

    summary_points = {
        "logit_sensitivity": beta0,
        "logit_specificity": spec_est,
        "sensitivity": summary_sens,
        "specificity": summary_spec,
        "sensitivity_percent": summary_sens_pct,
        "specificity_percent": summary_spec_pct,
        "ci_logit_sensitivity": ci_sens_logit,
        "ci_logit_specificity": ci_spec_logit,
        "ci_sensitivity": ci_sens,
        "ci_specificity": ci_spec,
        "ci_sensitivity_percent": ci_sens_pct,
        "ci_specificity_percent": ci_spec_pct
    }

    # Extract the estimated random effects covariance matrix
    # re_cov is a 2x2 matrix with:
    #   Var(u0) = variance for logit sensitivity (random intercept)
    #   Var(u1) = variance for the random slope (contributing to logit specificity)
    #   Cov(u0, u1) = covariance between them.
    re_cov = model.cov_re
    var_sens_re = re_cov.iloc[0, 0]
    var_spec_re = re_cov.iloc[0, 0] + re_cov.iloc[1, 1] + 2 * re_cov.iloc[0, 1]
    corr = re_cov.iloc[0, 1] / np.sqrt(re_cov.iloc[0, 0] * re_cov.iloc[1, 1])
    heterogeneity = {
        "var_logit_sensitivity": var_sens_re,
        "var_logit_specificity": var_spec_re,
        "correlation": corr
    }

    # --- Plotting ---
    # Plot individual studies (with study IDs) and overlay the summary point with error bars.
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        # x-axis: 1 - specificity; y-axis: sensitivity
        x = 1 - row['specificity']
        y = row['sensitivity']
        plt.scatter(x, y, color='blue', alpha=0.7)
        plt.annotate(row['study_id'], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Summary point: x = 1 - summary specificity, y = summary sensitivity
    sum_x = 1 - summary_spec
    sum_y = summary_sens

    # Compute approximate asymmetrical error bars on the probability scale
    x_err_lower = summary_spec - expit(ci_spec_logit[0])
    x_err_upper = expit(ci_spec_logit[1]) - summary_spec
    y_err_lower = summary_sens - expit(ci_sens_logit[0])
    y_err_upper = expit(ci_sens_logit[1]) - summary_sens

    plt.errorbar(sum_x, sum_y,
                 xerr=[[x_err_lower], [x_err_upper]],
                 yerr=[[y_err_lower], [y_err_upper]],
                 fmt='o', color='red', capsize=5, label='Summary (Bivariate)')

    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Bivariate Random-Effects Meta-Analysis with 95% CI')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    df_summary_points = pd.DataFrame(summary_points)
    return model, df_summary_points, heterogeneity

def bivariate_meta_analysis(df_study: pd.DataFrame):
    """
    Fit a bivariate random-effects meta-analysis model for sensitivity and specificity.
    The study's sample size (test_data) is used to weight observations by replicating rows.

    Parameters:
      df (pd.DataFrame): DataFrame with columns 'study_id', 'sensitivity', 'specificity', and 'test_data'

    Returns:
      model: Fitted mixed effects model
      summary_points: Dictionary with summary sensitivity and specificity estimates
      re_cov: Estimated random effects covariance matrix (between-study heterogeneity)
    """
    # Copy data and transform to logit scale
    df = df_study.copy()
    df['specificity'] = df['specificity'] / 100
    df['sensitivity'] = df['sensitivity'] / 100

    df['logit_sens'] = logit(df['sensitivity'])
    df['logit_spec'] = logit(df['specificity'])

    # Convert to long format: one row per outcome per study
    long_df = pd.DataFrame({
        'study_id': np.repeat(df['study_id'].values, 2),
        'measure': np.tile(['sensitivity', 'specificity'], len(df)),
        'logit_value': np.concatenate([df['logit_sens'].values, df['logit_spec'].values])
    })
    long_df['measure_num'] = (long_df['measure'] == 'specificity').astype(int)

    # Map the sample size from df into the long format
    long_df['test_data'] = long_df['study_id'].map(df.set_index('study_id')['test_data'])

    # Replicate rows according to the sample size
    long_df = long_df.loc[long_df.index.repeat(long_df['test_data'])].reset_index(drop=True)

    # Fit the bivariate mixed effects model.
    model = smf.mixedlm("logit_value ~ measure_num", long_df, groups=long_df["study_id"],
                        re_formula="~measure_num").fit()

    beta0 = model.params["Intercept"]
    beta1 = model.params["measure_num"]

    summary_logit_sens = beta0
    summary_logit_spec = beta0 + beta1

    summary_sens = expit(summary_logit_sens)
    summary_spec = expit(summary_logit_spec)

    summary_points = {
        "sensitivity": summary_sens,
        "specificity": summary_spec
    }

    # Extract the estimated random effects covariance matrix
    re_cov = model.cov_re  # a 2x2 matrix: [var(intercept), cov; cov, var(measure_num)]

    # Plot the summary point along with individual studies
    plt.figure(figsize=(8, 6))
    plt.scatter(1 - df['specificity'], df['sensitivity'], c='gray', alpha=0.7, label='Studies')
    plt.scatter(1 - summary_spec, summary_sens, c='blue', s=100, label='Summary (Bivariate)')

    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Bivariate Random-Effects Meta-Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, summary_points, re_cov


def plot_sroc_space(df_study:pd.DataFrame,
                    output_path:Optional[pathlib.Path],
                    title:str='ROC Space by Study ID',
                    file_name:str='sroc_space.png') -> None:
    # Calculate 1 - specificity (false positive rate)
    df = df_study.copy()
    df['one_minus_specificity'] = 100 - df['specificity']

    # Plotting ROC curve in the ROC space
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group data by study_id and plot each separately with a legend entry
    for study, group in df.groupby('study_id'):
        ax.plot(group['one_minus_specificity'], group['sensitivity'], marker='o', linestyle='-', label=study)

    ax.set_xlabel('1 - Specificity (False Positive Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title(title)
    ax.legend(title='Study ID')
    ax.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(file_name), dpi=300)
    plt.show()

