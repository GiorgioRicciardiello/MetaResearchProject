import numpy as np
import pandas as pd
from statsmodels.stats.meta_analysis import combine_effects
from config.config import config
import matplotlib.pyplot as plt
from typing import Optional
from scipy.special import expit, logit
import pathlib
from scipy.stats import norm


def compute_variance_proportion(proportion: float, n: int):
    """
    Compute the variance of a proportion using the binomial variance formula.
    Variance = p x (1-p)/n
    Parameters:
    -----------
    proportion : float
        The proportion (e.g., sensitivity or specificity) value.
    n : int
        The sample size used to compute the proportion. For sensitivity, this is the number
        of patients with the condition; for specificity, it is the number of patients without the condition.

    Returns:
    --------
    float
        The variance of the proportion.
    """
    if proportion != proportion:
        return proportion
    # Ensure the proportion is in the correct range
    if proportion < 0 or proportion > 1:
        raise ValueError("Proportion must be between 0 and 1.")
    if n <= 0:
        raise ValueError("Sample size must be positive.")
    return proportion * (1 - proportion) / n


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



def process_study(df_metrics:pd.DataFrame, threshold:int) -> pd.DataFrame:
    df = df_metrics.copy()
    df.rename(columns={f'Sensitivity_AHI_{threshold}': 'sensitivity',
                       f'Specificity_AHI_{threshold}': 'specificity'}, inplace=True)
    df.dropna(subset=['sensitivity', 'specificity', 'sample_size', 'study_id'], inplace=True)

    df['sens_prop'] = df['sensitivity']/100
    df['spec_prop'] = df['specificity']/100
    df['sens_variance'] = df.apply(lambda row: compute_variance_proportion(proportion=row['sens_prop'],
                                                                                n=row['sample_size']),
                                           axis=1)
    df['spec_variance'] = df.apply(lambda row: compute_variance_proportion(proportion=row['spec_prop'],
                                                                                n=row['sample_size']),
                                             axis=1)

    # Bound values to avoid issues with logit (e.g., exactly 0 or 1)
    eps = 1e-5
    df['sens_prop'] = df['sens_prop'].clip(eps, 1 - eps)
    df['spec_prop'] = df['spec_prop'].clip(eps, 1 - eps)
    # clip the variance to avoid division by zero
    df['sens_variance'] = df['sens_variance'].clip(lower=eps)
    df['spec_variance'] = df['spec_variance'].clip(lower=eps)

    return df

def meta_analysis_by_threshold(df_study:pd.DataFrame,
                               output_path:Optional[pathlib.Path]=None) -> pd.DataFrame:
    """
    Perform a meta-analysis for sensitivity and specificity for studies using a specified AHI threshold.

    This function uses the fixed-effect model from statsmodels' meta_analysis module to compute the
    pooled estimates for sensitivity and specificity, along with their 95% confidence intervals and
    heterogeneity statistics (Q and I²).

    Parameters
    ----------
    studies : list of dict
        List where each dictionary represents a study. Each study should include:
            - 'ahi_threshold': int, the AHI threshold (e.g., 5, 15, or 30)
            - 'sensitivity': float, the reported sensitivity
            - 'specificity': float, the reported specificity
            - 'se_sensitivity': float, the standard error of the sensitivity
            - 'se_specificity': float, the standard error of the specificity
    threshold : int
        The AHI threshold (5, 15, or 30) for which the meta-analysis is conducted.

    Returns
    -------
    results : dict
        Dictionary containing:
            - 'threshold': the AHI threshold analyzed
            - 'pooled_sensitivity': the pooled sensitivity estimate
            - 'ci_sensitivity': 95% confidence interval for sensitivity (lower, upper)
            - 'Q_sensitivity': Cochran's Q statistic for sensitivity
            - 'I2_sensitivity': I² statistic (heterogeneity) for sensitivity
            - 'pooled_specificity': the pooled specificity estimate
            - 'ci_specificity': 95% confidence interval for specificity (lower, upper)
            - 'Q_specificity': Cochran's Q statistic for specificity
            - 'I2_specificity': I² statistic (heterogeneity) for specificity

    Raises
    ------
    ValueError
        If no studies are found for the specified AHI threshold.
    """

    # df_study = df_studies.copy()
    alpha = 0.05
    # Filter studies by the specified AHI threshold and that have both measures
    # df_study.rename(columns={f'Sensitivity_AHI_{threshold}':'sensitivity',
    #                          f'Specificity_AHI_{threshold}':'specificity'}, inplace=True)
    #
    # df_study.dropna(subset=['sensitivity', 'specificity'], inplace=True)
    # ---------------------------
    # Meta-analysis for Sensitivity
    # ---------------------------
    # # Extract sensitivity values and calculate their variances (square of standard errors)
    # df_study['sens_prop'] = df_study['sensitivity']/100
    # df_study['spec_prop'] = df_study['specificity']/100
    # df_study['sens_variance'] = df_study.apply(lambda row: compute_variance_proportion(proportion=row['sens_prop'],
    #                                                                             n=row['sample_size']),
    #                                        axis=1)
    # df_study['spec_variance'] = df_study.apply(lambda row: compute_variance_proportion(proportion=row['spec_prop'],
    #                                                                             n=row['sample_size']),
    #                                          axis=1)
    #
    # # Bound values to avoid issues with logit (e.g., exactly 0 or 1)
    # eps = 1e-5
    # df_study['sens_prop'] = df_study['sens_prop'].clip(eps, 1 - eps)
    # df_study['spec_prop'] = df_study['spec_prop'].clip(eps, 1 - eps)
    # # clip the variance to avoid divsion by zero
    # df_study['sens_variance'] = df_study['sens_variance'].clip(lower=eps)
    # df_study['spec_variance'] = df_study['spec_variance'].clip(lower=eps)

    df_effects = pd.DataFrame()
    for metric in ['sens', 'spec']:
        # metric = 'sens'
        # Use statsmodels combine_effects to pool the sensitivity estimates using a fixed-effect model
        CombineResults = combine_effects(effect=df_study[f'{metric}_prop'],
                                         variance=df_study[f'{metric}_variance'],
                                         method_re="iterated",
                                         use_t=False,
                                         alpha=alpha,
                                         row_names=df_study.study_id)

        # CombineResults.conf_int(alpha=alpha)  # confidence interval for the overall mean estimate
        # CombineResults.conf_int_samples(alpha=alpha)  # confidence intervals for the effect size estimate of samples
        if output_path:
            fig_se = CombineResults.plot_forest(alpha=alpha)  # Forest plot with means and confidence intervals
            fig_se.savefig(output_path.joinpath(f'forest_plot_{metric}.png'))
            plt.show()

        # CombineResults.summary_array(alpha=alpha)  # Create array with sample statistics and mean estimates
        df_combined_effects = CombineResults.summary_frame()  # Create DataFrame with sample statistics and mean estimates
        # CombineResults.test_homogeneity()  # Test whether the means of all samples are the same
        try:
            pooled_effect, Q, I2 = compute_Q_and_I2(
                effects=df_study[f'{metric}_prop'],
                variances=df_study[f'{metric}_variance']
            )
        except Exception:
            pooled_effect, Q, I2 = np.nan, np.nan,np.nan

        df_combined_effects['metric'] = metric
        df_combined_effects['Q'] = Q if Q is not None else np.nan
        df_combined_effects['I2'] = I2 if I2 is not None else np.nan
        df_combined_effects['pooled_effect'] = pooled_effect
        df_effects = pd.concat([df_effects, df_combined_effects], axis=0)
    df_effects['threshold'] = thr

    return df_effects


def compute_forest_plot_dor(df_studies: pd.DataFrame,
                            threshold: int,
                            output_path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    Compute diagnostic odds ratios (DOR) from sensitivity and specificity for a given AHI threshold,
    and generate a forest plot of the log DOR with 95% confidence intervals.

    The DOR is computed as:
        DOR = (sensitivity/(1-sensitivity)) / ((1-specificity)/specificity)
            = (sensitivity * specificity) / ((1-sensitivity)*(1-specificity))
    which implies:
        log(DOR) = logit(sensitivity) + logit(specificity)

    We assume sensitivity and specificity are provided as percentages in the input DataFrame.
    The approximate variance for the logit-transformed sensitivity and specificity is computed as:
        Var(logit(p)) ≈ 1 / (n * p * (1 - p))
    where `n` is the sample size. (Note: ideally, the number of diseased and non-diseased individuals
    would be used for sensitivity and specificity respectively; here, sample_size is used as an approximation.)

    Parameters
    ----------
    studies : pd.DataFrame
        DataFrame containing study-level diagnostic metrics. Expected columns:
            - 'Sensitivity_AHI_{threshold}': Sensitivity (in percentage) for the specified threshold.
            - 'Specificity_AHI_{threshold}': Specificity (in percentage) for the specified threshold.
            - 'sample_size': Total sample size used in the study (approximation for variance calculations).
            - 'study_id': Identifier for the study.
    threshold : int
        The AHI threshold to analyze.
    output_path : Optional[pathlib.Path], optional
        Directory to save the forest plot image; if None, the plot is only shown.

    Returns
    -------
    df_dor : pd.DataFrame
        DataFrame with the study ID, DOR, and 95% confidence intervals for the DOR.
    """
    # Create a working copy of the DataFrame and rename the relevant columns

    df = process_study(df_metrics=df_studies, threshold=threshold)

    # Compute logit transformations
    df['logit_sens'] = np.log(df['sens_prop'] / (1 - df['sens_prop']))
    df['logit_spec'] = np.log(df['spec_prop'] / (1 - df['spec_prop']))

    # Compute log diagnostic odds ratio
    df['log_dor'] = df['logit_sens'] + df['logit_spec']
    df['DOR'] = np.exp(df['log_dor'])

    # Compute approximate variances for logit values using sample_size as proxy.
    # Note: This is an approximation; ideally, one would use the actual number of disease positives and negatives.
    df['var_logit_sens'] = 1 / (df['sample_size'] * df['sens_prop'] * (1 - df['sens_prop']))
    df['var_logit_spec'] = 1 / (df['sample_size'] * df['spec_prop'] * (1 - df['spec_prop']))
    df['var_log_dor'] = df['var_logit_sens'] + df['var_logit_spec']

    # Compute 95% confidence intervals for log_dor
    z = 1.96  # approximate z-score for 95% CI
    df['ci_lower_log'] = df['log_dor'] - z * np.sqrt(df['var_log_dor'])
    df['ci_upper_log'] = df['log_dor'] + z * np.sqrt(df['var_log_dor'])

    # Back-transform to obtain CI for DOR
    df['ci_lower'] = np.exp(df['ci_lower_log'])
    df['ci_upper'] = np.exp(df['ci_upper_log'])

    # Prepare forest plot
    plt.figure(figsize=(8, len(df) * 0.6 + 2))
    y_positions = np.arange(len(df))

    # Plot horizontal lines for CI and points for the estimated DOR
    plt.errorbar(df['DOR'], y_positions, xerr=[df['DOR'] - df['ci_lower'], df['ci_upper'] - df['DOR']],
                 fmt='o', color='black', ecolor='gray', capsize=3)

    plt.yticks(y_positions, df['study_id'])
    plt.xlabel('Diagnostic Odds Ratio (DOR)')
    plt.title(f'Forest Plot of Diagnostic Odds Ratios for AHI Threshold {threshold}')
    plt.axvline(x=1, color='red', linestyle='--', label='No Diagnostic Discrimination')
    plt.legend()
    plt.gca().invert_yaxis()  # invert y-axis so that the first study is at the top

    if output_path:
        plot_file = output_path.joinpath(f'forest_plot_dor_threshold_{threshold}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return a DataFrame summarizing the DOR results
    df_dor = df[['study_id', 'DOR', 'ci_lower', 'ci_upper']].copy()
    df_dor.reset_index(drop=True, inplace=True)

    return df_dor


def compute_sroc_by_threshold(studies: pd.DataFrame,
                              threshold: int,
                              output_path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    Compute the Summary Receiver Operating Characteristic (sROC) curve for a diagnostic test meta-analysis.

    This function implements a Moses-Littenberg approach:
      1. Converts sensitivity and specificity (provided as percentages) to proportions.
      2. Computes the false positive rate (FPR = 1 - specificity).
      3. Applies the logit transformation to both sensitivity (TPR) and FPR.
      4. Calculates D = logit(TPR) - logit(FPR) and S = logit(TPR) + logit(FPR) for each study.
      5. Performs linear regression: D = a + b * S.
      6. Generates the sROC curve using:
             TPR = expit(a + b * log(FPR/(1 - FPR)))
         over a range of FPR values.
      7. Computes an approximate AUC using AUC = Φ(a / sqrt(1 + b²)).

    Parameters
    ----------
    studies : pd.DataFrame
        DataFrame containing study-level diagnostic accuracy metrics. Must include columns:
            - 'Sensitivity_AHI_{threshold}': Sensitivity (in percentage) for the specified threshold.
            - 'Specificity_AHI_{threshold}': Specificity (in percentage) for the specified threshold.
    threshold : int
        The AHI threshold to filter studies.
    output_path : Optional[pathlib.Path], optional
        If provided, the sROC plot will be saved to this directory.

    Returns
    -------
    sroc_results : pd.DataFrame
        DataFrame containing the sROC curve points (FPR and TPR) along with constant columns for
        the regression intercept (a), slope (b), and the computed AUC.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = studies.copy()
    # Rename columns for consistency
    df.rename(columns={f'Sensitivity_AHI_{threshold}': 'sensitivity',
                       f'Specificity_AHI_{threshold}': 'specificity'}, inplace=True)

    # Drop rows with missing sensitivity or specificity
    df.dropna(subset=['sensitivity', 'specificity'], inplace=True)

    # Convert percentages to proportions
    df['sens_prop'] = df['sensitivity'] / 100.0
    df['spec_prop'] = df['specificity'] / 100.0

    # Clip proportions to avoid issues with logit transformation (avoid exactly 0 or 1)
    eps = 1e-5
    df['sens_prop'] = df['sens_prop'].clip(eps, 1 - eps)
    df['spec_prop'] = df['spec_prop'].clip(eps, 1 - eps)

    # Compute False Positive Rate (FPR)
    df['fpr'] = 1 - df['spec_prop']
    df['fpr'] = df['fpr'].clip(eps, 1 - eps)

    # Compute the logit transformations for sensitivity and FPR
    df['logit_sens'] = np.log(df['sens_prop'] / (1 - df['sens_prop']))
    df['logit_fpr'] = np.log(df['fpr'] / (1 - df['fpr']))

    # Calculate D and S for each study
    df['D'] = df['logit_sens'] - df['logit_fpr']
    df['S'] = df['logit_sens'] + df['logit_fpr']

    # Perform linear regression: D = a + b * S using numpy.polyfit (returns [b, a])
    coeffs = np.polyfit(df['S'], df['D'], 1)
    b = coeffs[0]
    a = coeffs[1]

    # Generate sROC curve: for a range of FPR values, compute corresponding TPR using:
    # TPR = expit(a + b * log(FPR/(1-FPR)))
    fpr_values = np.linspace(eps, 1 - eps, 100)
    logit_fpr_values = np.log(fpr_values / (1 - fpr_values))
    tpr_values = expit(a + b * logit_fpr_values)

    # Compute approximate AUC using: AUC = norm.cdf(a / sqrt(1 + b^2))
    auc = norm.cdf(a / np.sqrt(1 + b ** 2))

    # Assemble the sROC curve DataFrame
    sroc_df = pd.DataFrame({
        'FPR': fpr_values,
        'TPR': tpr_values
    })
    sroc_df['a'] = a
    sroc_df['b'] = b
    sroc_df['AUC'] = auc

    # Plot the sROC curve if an output path is provided
    if output_path:
        plt.figure(figsize=(8, 6))
        plt.plot(sroc_df['FPR'], sroc_df['TPR'], label=f'sROC (AUC={auc:.2f})', lw=2)
        # Plot the individual study points
        plt.scatter(1 - df['spec_prop'], df['sens_prop'], color='red', label='Study Points', zorder=5)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'sROC Curve for AHI Threshold {threshold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path.joinpath(f'sroc_curve_threshold_{threshold}.png'))
        plt.close()

    return sroc_df


if __name__ == "__main__":
    # Example list of studies.
    # Each study includes:
    #   - 'ahi_threshold': AHI threshold used in the study
    #   - 'sensitivity': Reported sensitivity value
    #   - 'specificity': Reported specificity value
    #   - 'se_sensitivity': Standard error for sensitivity
    #   - 'se_specificity': Standard error for specificity
    df_studies = pd.read_excel(config.get('papers_manually_selected'))

    col_static = ['study_id',
                  # 'study_name',
                  'sample_size', 'sample_train', 'sample_test']
    # AHI thresholds of interest
    thresholds = [5, 15, 30]

    # Perform meta-analysis for each threshold and print the results
    df_meta_analysis = pd.DataFrame(columns=col_static)
    for thr in thresholds:
        # thr = thresholds[0]
        col_metrics = [col for col in df_studies if f'_{thr}' in col]
        col_subset = col_static + col_metrics
        df_subsets = df_studies[col_subset].copy()
        df_study = process_study(df_metrics=df_subsets, threshold=thr)

        df_result = meta_analysis_by_threshold(df_study=df_study,
                                               threshold=thr,
                                               output_path=None)

        df_result = meta_analysis_by_threshold_corrected(df_studies=df_subsets,
                                               threshold=thr,
                                               output_path=None)
        df_result['AHI'] = thr
        df_meta_analysis = pd.concat([df_meta_analysis, df_result], axis=1)


