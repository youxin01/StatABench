from mcp.server.fastmcp import FastMCP
import argparse
import os
import pandas as pd
from typing import Union, List
import os
import math
import numpy as np
import pandas as pd

from typing import Union, List, Dict, Optional, Literal

from pandas.api.types import is_numeric_dtype

import scipy.stats as stats
from scipy.stats import (
    chi2_contingency, fisher_exact, mood, mannwhitneyu, wilcoxon, kruskal, ttest_ind, norm
)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LassoCV, RidgeCV, ElasticNetCV, BayesianRidge, LogisticRegression
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import NearestNeighbors

import pingouin as pg
from pmdarima import auto_arima

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

import pymc as pm
import arviz as az
from cvxpy import Variable, Minimize, Problem, sum, quad_form

from sklearn.preprocessing import LabelEncoder

mcp = FastMCP("Modeling Problem")

@mcp.tool()
def calculate_statistic(data: Union[pd.DataFrame, str], column: str, method: str) -> Union[float, int, str, List[float]]:
    """
    Calculate a specified statistic for a given column in a DataFrame or CSV file.

    Parameters:
        data (Union[pd.DataFrame, str]): The input data, either as a pandas DataFrame or a CSV file path.
        column (str): The column name to perform the statistic on.
        method (str): The statistical method to apply. Supported methods (case-insensitive):
                      - 'range' : max - min
                      - 'quartile' : returns [Q1, Q2 (median), Q3]
                      - 'kurtosis' : measure of tailedness
                      - 'skewness' : measure of asymmetry
                      - 'mean' : arithmetic average
                      - 'median' : middle value
                      - 'mode' : most frequent value(s); returns value if one mode, or list if multiple
                      - 'standard_deviation' : sample standard deviation

    Returns:
        Union[float, int, str, List[float]]: The calculated statistic.
    """
    
    # Load data
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path '{data}' was not found.")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Input 'data' must be a pandas DataFrame or a file path.")
    
    # Validate column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    col_data = df[column]
    
    # Normalize method
    method = method.strip().lower()
    
    if method == 'range':
        return col_data.max() - col_data.min()
    elif method == 'quartile':
        return [col_data.quantile(q) for q in [0.25, 0.5, 0.75]]
    elif method == 'kurtosis':
        return col_data.kurt()
    elif method == 'skewness':
        return col_data.skew()
    elif method == 'mean':
        return col_data.mean()
    elif method == 'median':
        return col_data.median()
    elif method == 'mode':
        mode_values = col_data.mode()
        return mode_values.tolist() if len(mode_values) > 1 else mode_values.iloc[0]
    elif method == 'standard_deviation':
        return col_data.std()
    else:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are: "
                         "'range', 'quartile', 'kurtosis', 'skewness', 'mean', "
                         "'median', 'mode', 'standard_deviation'.")

@mcp.tool()
def check_missing_values(
    data: Union[str, pd.DataFrame],
    columns: Union[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Check missing values in the specified columns of a dataset.

    Parameters:
        data (Union[str, pd.DataFrame]): The dataset, either as a pandas DataFrame
                                         or a string path to a CSV file.
        columns (Union[str, List[str]]): The column name(s) to check. Can be a single
                                         column name (str) or a list of column names.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a column name, and the value
                                     is another dictionary containing:
                                     - 'missing_count': Number of missing (NaN) entries.
                                     - 'missing_percentage': Percentage of missing entries (rounded to 2 decimals).
    """
    # Load data if given as a file path
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path '{data}' was not found.")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input 'data' must be a pandas DataFrame or a file path.")

    # Normalize columns to a list
    columns_to_check = [columns] if isinstance(columns, str) else columns

    # Check that all columns exist
    missing_cols = [col for col in columns_to_check if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found in DataFrame: {', '.join(missing_cols)}")

    # Calculate missing values
    results = {}
    total_rows = len(df)
    for col in columns_to_check:
        missing_count = int(df[col].isnull().sum())
        missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0.0
        results[col] = {
            'missing_count': missing_count,
            'missing_percentage': round(missing_percentage, 2)
        }

    return results

@mcp.tool()
def detect_outliers(
    data: Union[str, pd.DataFrame],
    columns: Union[str, List[str]],
    iqr_multiplier: float = 1.5
) -> Dict[str, Dict[str, List]]:
    """
    Detect outliers in specified numeric columns using the IQR method.

    Parameters:
        data: DataFrame or CSV file path.
        columns: Column name or list of column names to check.
        iqr_multiplier: Multiplier for the IQR to define outlier bounds (default 1.5).

    Returns:
        Dictionary mapping column names to dictionaries with outlier indices and values.
    """
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    columns_to_check = [columns] if isinstance(columns, str) else columns

    for col in columns_to_check:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if not is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric for outlier detection.")

    results = {}
    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        results[col] = {
            'indices': df[mask].index.tolist(),
            'values': df.loc[mask, col].tolist()
        }
    return results

@mcp.tool()
def check_column_type_is(
    data: Union[str, pd.DataFrame],
    columns: Union[str, List[str]],
    target_type: str
) -> Dict[str, bool]:
    """
    Check if specified columns are of the target type.

    Parameters:
        data: DataFrame or CSV file path.
        columns: Single column name or list of column names to check.
        target_type: One of ['numeric', 'object', 'datetime'] indicating the type to check.

    Returns:
        Dict mapping column names to boolean indicating if they match the target type.
    """
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    columns_to_check = [columns] if isinstance(columns, str) else columns

    for col in columns_to_check:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    def is_type(series: pd.Series, target: str) -> bool:
        if target == 'numeric':
            return pd.api.types.is_numeric_dtype(series)
        elif target == 'object':
            return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        elif target == 'datetime':
            return pd.api.types.is_datetime64_any_dtype(series)
        else:
            raise ValueError(f"Unsupported target_type: {target}")

    return {col: is_type(df[col], target_type) for col in columns_to_check}

@mcp.tool()
def correlation_analysis(
    data: Union[pd.DataFrame, str],
    method: str,
    columns: Optional[List[str]] = None,
    covar: Optional[Union[str, List[str]]] = None
):
    """
    Perform correlation analysis on the given dataset.

    Parameters:
        data (Union[pd.DataFrame, str]): The input data, either as a pandas DataFrame or a CSV file path.
        method (str): Correlation method to use (case-insensitive). Supported:
                      - 'pearson' : linear correlation
                      - 'kendall' : rank correlation
                      - 'partial' : partial correlation (requires 'covar')
        columns (Optional[List[str]]): List of columns to include in analysis. Defaults to all numeric columns.
        covar (Optional[Union[str, List[str]]]): Covariate(s) to control for in partial correlation.
                                                 Required if method is 'partial'.

    Returns:
        pd.DataFrame: Correlation matrix.

    Raises:
        FileNotFoundError: If CSV file path is invalid.
        TypeError: If data is neither DataFrame nor file path.
        ValueError: If columns are invalid or method is unsupported.
    """

    # Load data
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path '{data}' was not found.")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input 'data' must be a pandas DataFrame or a CSV file path.")

    # Select subset of data
    if columns:
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")
        df_subset = df[columns]
    else:
        df_subset = df

    df_numeric = df_subset.select_dtypes(include='number')

    if df_numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required for correlation analysis.")

    method = method.strip().lower()

    if method in ['pearson', 'kendall']:
        return df_numeric.corr(method=method)

    elif method == 'partial':
        import pingouin as pg

        if covar is None:
            raise ValueError("The 'covar' argument is required for partial correlation.")

        covar_list = [covar] if isinstance(covar, str) else covar
        for c in covar_list:
            if c not in df.columns:
                raise ValueError(f"Covariate column '{c}' not found in the DataFrame.")

        # Variables to analyze = numeric columns excluding covariates
        variables = [col for col in df_numeric.columns if col not in covar_list]
        if len(variables) < 2:
            raise ValueError("At least two numeric columns (excluding covariates) are required.")

        result = pd.DataFrame(index=variables, columns=variables, dtype=float)

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i > j:
                    continue
                if var1 == var2:
                    result.at[var1, var2] = 1.0
                else:
                    r = pg.partial_corr(data=df, x=var1, y=var2, covar=covar_list)['r'].iloc[0]
                    result.at[var1, var2] = r
                    result.at[var2, var1] = r

        return result

    else:
        raise ValueError("Unsupported method '{}'. Please choose from 'pearson', 'kendall', or 'partial'.".format(method))

@mcp.tool()
def ci_normal(param_to_estimate, sample_mean, sample_std, n, confidence=0.95, 
              population_std=None, population_mean=None):
    """
    Compute confidence interval for normal population parameters.

    Parameters:
    - param_to_estimate: 'mean' or 'variance'
    - sample_mean: float, sample mean
    - sample_std: float, sample standard deviation
    - n: int, sample size
    - confidence: float, confidence level (default 0.95)
    - population_std: float or None, known population standard deviation
    - population_mean: float or None, known population mean

    Returns:
    - (lower_bound, upper_bound): tuple of floats
    """
    alpha = 1 - confidence

    if param_to_estimate == 'mean':
        if population_std is not None:
            # σ known: use Z-distribution
            z = stats.norm.ppf(1 - alpha / 2)
            margin = z * population_std / math.sqrt(n)
            return sample_mean - margin, sample_mean + margin
        else:
            # σ unknown: use t-distribution
            t = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t * sample_std / math.sqrt(n)
            return sample_mean - margin, sample_mean + margin

    elif param_to_estimate == 'variance':
        sample_variance = sample_std ** 2
        df = n if population_mean is not None else n - 1

        chi2_lower = stats.chi2.ppf(alpha / 2, df=df)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=df)

        lower = df * sample_variance / chi2_upper
        upper = df * sample_variance / chi2_lower
        return lower, upper

    else:
        raise ValueError("param_to_estimate must be 'mean' or 'variance'")

@mcp.tool()
def ci_two_normal(data_type, 
                  sample_mean1: float = None, sample_std1: float = None, n1: int = None, 
                  sample_mean2: float = None, sample_std2: float = None, n2: int = None,
                  confidence: float = 0.95, 
                  population_std_known: bool = False, 
                  assume_equal_variance: bool = False, 
                  population_mean_known: bool = False):
    """
    Compute confidence interval for two-sample normal distributions: difference in means or ratio of variances.

    Parameters:
    - data_type: 'mean_diff' or 'var_ratio'
    - sample_mean1, sample_std1, n1: sample stats for group 1
    - sample_mean2, sample_std2, n2: sample stats for group 2
    - confidence: confidence level (e.g., 0.95)
    - population_std_known: if True, use Z-distribution for mean_diff
    - assume_equal_variance: assume equal variances when population std unknown
    - population_mean_known: for var_ratio, affects degrees of freedom

    Returns:
    - (lower, upper): confidence interval tuple
    """
    alpha = 1 - confidence

    if data_type == 'mean_diff':
        diff = sample_mean1 - sample_mean2
        var1 = sample_std1 ** 2
        var2 = sample_std2 ** 2

        if population_std_known:
            # σ1, σ2 known ⇒ Z-distribution
            se = math.sqrt(var1 / n1 + var2 / n2)
            z = stats.norm.ppf(1 - alpha / 2)
            margin = z * se
            return diff - margin, diff + margin

        else:
            # σ unknown ⇒ t-distribution
            if assume_equal_variance:
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                se = math.sqrt(pooled_var * (1 / n1 + 1 / n2))
                df = n1 + n2 - 2
            else:
                se = math.sqrt(var1 / n1 + var2 / n2)
                df_num = (var1 / n1 + var2 / n2) ** 2
                df_denom = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
                df = df_num / df_denom
            t = stats.t.ppf(1 - alpha / 2, df)
            margin = t * se
            return diff - margin, diff + margin

    elif data_type == 'var_ratio':
        df1 = n1 if population_mean_known else n1 - 1
        df2 = n2 if population_mean_known else n2 - 1
        f_lower = stats.f.ppf(alpha / 2, dfn=df1, dfd=df2)
        f_upper = stats.f.ppf(1 - alpha / 2, dfn=df1, dfd=df2)
        ratio = (sample_std1 ** 2) / (sample_std2 ** 2)
        return ratio / f_upper, ratio / f_lower

    else:
        raise ValueError("data_type must be 'mean_diff' or 'var_ratio'")

@mcp.tool()
def contingency_test(data, columns=None, method='chi-square independence'):
    """
    Perform contingency table tests for association between categorical variables.

    Parameters:
        data: pandas DataFrame or path to CSV file.
        columns: List of 2 or 3 column names. For 2 columns: row and column variables.
                 For 3 columns: row, column, and stratification variable (for Mantel-Haenszel).
        method: Statistical test to use. Options:
                - 'chi-square independence'
                - 'fisher exact' (only for 2x2 tables)
                - 'mantel-haenszel' (requires 3 columns: row, col, stratification)

    Returns:
        dict: Test results including statistics, p-values, and confidence intervals (if applicable).
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = None  

    method = method.lower()

    if method == "chi-square independence":
        if df is None:
            table = np.array(data)
        else:
            row, col = columns
            table = pd.crosstab(df[row], df[col]).to_numpy()
        stat, p, dof, expected = chi2_contingency(table)
        return {"method": "Chi-square Independence", "statistic": stat, "p_value": p, "dof": dof}

    elif method == "fisher exact":
        if df is None:
            table = np.array(data)
        else:
            row, col = columns
            table = pd.crosstab(df[row], df[col]).to_numpy()
        if table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test only supports 2x2 tables.")
        odds, p = fisher_exact(table)
        return {"method": "Fisher's Exact Test", "odds_ratio": odds, "p_value": p}

    elif method == "mantel-haenszel":
        if df is None:
            raise ValueError("Mantel-Haenszel 需要原始数据（DataFrame/CSV）。")
        if len(columns) != 3:
            raise ValueError("Mantel-Haenszel 需要 [row, col, stratify] 三列。")

        row, col, strat = columns
        tables = []
        for _, g in df.groupby(strat):
            t = pd.crosstab(g[row], g[col])
            if t.shape != (2, 2):
                raise ValueError("每个分层必须是 2x2 列联表。")
            tables.append(t.to_numpy())

        result = StratifiedTable(tables)
        return {
            "method": "Mantel-Haenszel",
            "common_odds_ratio": result.oddsratio_pooled,
            "ci95": result.oddsratio_pooled_confint(),
            "p_value": result.test_null_odds().pvalue,
        }

    else:
        raise ValueError(f"Unknown method: {method}")


@mcp.tool()
def ks_test(
    data,
    columns,
    mode='one-sample',
    dist='norm',
    alternative='two-sided'
):
    """
    Perform Kolmogorov-Smirnov test.

    Parameters:
        data: pandas DataFrame or CSV file path.
        columns: List of column names to test (1 for one-sample, 2 for two-sample).
        mode: 'one-sample' or 'two-sample' (default: 'one-sample').
        dist: Theoretical distribution name for one-sample test (e.g., 'norm', 'expon').
        alternative: Defines the alternative hypothesis ('two-sided', 'less', 'greater').

    Returns:
        dict: Test results including statistic and p-value.
    """
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File '{data}' does not exist.")
        df = pd.read_csv(data)
    else:
        df = data.copy()

    if mode == 'one-sample':
        if len(columns) != 1:
            raise ValueError("One-sample KS test requires exactly one column.")
        x = df[columns[0]].dropna()
        dist_func = getattr(stats, dist, None)
        if dist_func is None:
            raise ValueError(f"Unsupported distribution: {dist}")
        dist_obj = getattr(stats, dist)
        dist_args = dist_obj.fit(x)  # auto-fit distribution parameters
        print(f"Estimated parameters for {dist}: {dist_args}")

        stat, p = stats.kstest(x, dist, args=dist_args, alternative=alternative)
        return {
            'test': 'Kolmogorov-Smirnov one-sample',
            'column': columns[0],
            'distribution': dist,
            'statistic': stat,
            'p_value': p
        }

    elif mode == 'two-sample':
        if len(columns) != 2:
            raise ValueError("Two-sample KS test requires exactly two columns.")
        x1 = df[columns[0]].dropna()
        x2 = df[columns[1]].dropna()
        stat, p = stats.ks_2samp(x1, x2, alternative=alternative)
        return {
            'test': 'Kolmogorov-Smirnov two-sample',
            'columns': columns,
            'statistic': stat,
            'p_value': p
        }

    else:
        raise ValueError("Parameter 'mode' must be 'one-sample' or 'two-sample'.")

@mcp.tool()
def simple_linear_regression(data, x_col: str, y_col: str) -> dict:
    """
    Perform simple linear regression on two columns.

    Parameters:
        df (pd.DataFrame): The data frame containing the data.
        x_col (str): Name of the predictor (independent variable).
        y_col (str): Name of the response (dependent variable).

    Returns:
        dict: A dictionary containing the regression model, coefficients, R², p-value, and summary.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    x = df[x_col]
    y = df[y_col]

    # 添加常数项
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()

    return {
        "model": model,
        "intercept": model.params['const'],
        "coefficient": model.params[x_col],
        "r_squared": model.rsquared,
        "p_value": model.pvalues[x_col],
        "summary": model.summary().as_text()
    }

@mcp.tool()
def mood_variance_test(data, columns=None):
    """
    Perform Mood's test for equality of variances between two columns.

    Parameters:
        data: pandas DataFrame or CSV file path.
        columns: List or tuple of exactly two column names to compare variances.

    Returns:
        dict: Test statistic and p-value.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    if columns is None or len(columns) != 2:
        raise ValueError("Please provide exactly two column names for variance comparison.")

    x = data[columns[0]].dropna()
    y = data[columns[1]].dropna()

    stat, p = mood(x, y)

    return {
        "method": "Mood's Variance Test",
        "statistic": stat,
        "p_value": p
    }

@mcp.tool()
def multivariable_linear_regression(data, y_col, x_cols, show_vif=True):
    """
    Perform multivariable linear regression.

    Parameters:
        data (str or pd.DataFrame): Input data as a CSV file path or DataFrame.
        y_col (str): Name of the dependent variable column.
        x_cols (list of str): List of independent variable column names.
        show_vif (bool): Whether to calculate and return VIF (Variance Inflation Factor) for multicollinearity check (default True).

    Returns:
        dict: A dictionary containing the regression model, coefficients, p-values,
              R², adjusted R², and VIF (if requested).
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    X = df[x_cols]
    y = df[y_col]

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype).startswith("category")]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = X.astype({col: int for col in X.select_dtypes(include=['bool']).columns})
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # 添加常数项
    X = sm.add_constant(X)

    # 拟合OLS模型
    model = sm.OLS(y, X).fit()

    # 提取系数和显著性
    coefficients = model.params.to_dict()
    pvalues = model.pvalues.to_dict()

    # 提取模型拟合优度
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj

    # 提取VIF
    vif_dict = None
    if show_vif:
        vif_dict = {}
        for i, var in enumerate(X.columns):
            vif_dict[var] = variance_inflation_factor(X.values, i)

    return {
        "model": model,
        "coefficients": coefficients,
        "pvalues": pvalues,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "vif": vif_dict
    }

@mcp.tool()
def run_glm(
    data, 
    y_col, 
    x_cols, 
    method='gaussian'
):
    """
    Build and fit a Generalized Linear Model (GLM).

    Parameters:
        data (str or pd.DataFrame): Input data as a CSV file path or a DataFrame.
        y_col (str): Name of the dependent variable.
        x_cols (list of str): List of independent variable names.
        method (str): Distribution family to use for GLM.
                      Supported options: 'gaussian', 'logistic', 'poisson', 'gamma', 
                      'negativebinomial', 'inverse_gaussian', 'multinomial'.
                      Default is 'gaussian'.

    Returns:
        dict: A dictionary containing the regression model, coefficients, standard errors, p-values,
              AIC, BIC, and pseudo R² (if applicable).
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    data = data.dropna(subset=[y_col] + x_cols)
    if data[y_col].dtype == 'object' or data[y_col].dtype.name == 'category':
        le = LabelEncoder()
        data[y_col] = le.fit_transform(data[y_col])
    rename_map = {
        col: col.strip().replace(" ", "_").replace("-", "_")
        for col in data.columns
    }
    data = data.rename(columns=rename_map)
    y_col = rename_map.get(y_col, y_col)
    x_cols = [rename_map.get(col, col) for col in x_cols]

    formula = f"{y_col} ~ {' + '.join(x_cols)}"

    # Map method name to statsmodels family object
    method_map = {
        'gaussian': sm.families.Gaussian(),
        'logistic': sm.families.Binomial(),
        'poisson': sm.families.Poisson(),
        'gamma': sm.families.Gamma(),
        'negativebinomial': sm.families.NegativeBinomial(),
        'inverse_gaussian': sm.families.InverseGaussian(link=sm.families.links.log()),
    }

    if method.lower() == 'multinomial':
        # Use MNLogit for multinomial logistic regression
        model = smf.mnlogit(formula, data=data)
        result = model.fit()
    else:
        fam = method_map.get(method.lower())
        if fam is None:
            raise ValueError(f"Unsupported method: {method}")
        model = smf.glm(formula=formula, data=data, family=fam)
        result = model.fit()

    # Build structured dict
    output = {
        "method": method.lower(),
        "model": model,
        "coefficients": result.params.to_dict(),
        "std_errors": result.bse.to_dict(),
        "pvalues": result.pvalues.to_dict(),
        "aic": result.aic if hasattr(result, "aic") else None,
        "bic": result.bic if hasattr(result, "bic") else None,
        "pseudo_r2": getattr(result, "prsquared", None)  # available in logistic models
    }

    return output

@mcp.tool()
def nonparametric_test(data, columns, method='mannwhitney'):
    """
    Unified nonparametric test function.

    Parameters:
        data (str or pd.DataFrame): Input data as CSV file path or DataFrame.
        columns (list of str): Columns to test (2 for Mann–Whitney or Wilcoxon, 2+ for Kruskal).
        method (str): Test method: 'mannwhitney', 'wilcoxon', or 'kruskal'.

    Returns:
        dict: Test results including statistic and p-value.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    if method == 'mannwhitney':
        assert len(columns) == 2, "Mann–Whitney requires exactly 2 columns."
        x = data[columns[0]].dropna()
        y = data[columns[1]].dropna()
        stat, p = mannwhitneyu(x, y, alternative='two-sided')
        return {"method": "Mann–Whitney U", "statistic": stat, "p_value": p}

    elif method == 'wilcoxon':
        assert len(columns) == 2, "Wilcoxon requires exactly 2 columns."
        x = data[columns[0]].dropna()
        y = data[columns[1]].dropna()
        min_len = min(len(x), len(y))
        stat, p = wilcoxon(x.iloc[:min_len], y.iloc[:min_len])
        return {"method": "Wilcoxon Signed-Rank", "statistic": stat, "p_value": p}

    elif method == 'kruskal':
        assert len(columns) >= 2, "Kruskal-Wallis requires 2 or more columns."
        groups = [data[col].dropna() for col in columns]
        stat, p = kruskal(*groups)
        return {"method": "Kruskal–Wallis", "statistic": stat, "p_value": p}

    else:
        raise ValueError(f"Unsupported method: {method}")

@mcp.tool()
def huber_regression(data, y_col, x_cols):
    """
    Fit a Huber robust linear regression model.

    Parameters:
        data (str or pd.DataFrame): Input data as CSV file path or DataFrame.
        y_col (str): Name of the target (dependent) variable.
        x_cols (list of str): List of predictor (independent) variable names.

    Returns:
        dict: A dictionary containing the regression model, coefficients, standard errors, p-values,
              scale, and convergence status.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Drop rows with missing values in relevant columns
    df = data[x_cols + [y_col]].dropna()

    X = sm.add_constant(df[x_cols])
    y = df[y_col]

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    result = model.fit()

    output = {
        "method": "huberregression",
        "model": model,
        "coefficients": result.params.to_dict(),
        "std_errors": result.bse.to_dict(),
        "pvalues": result.pvalues.to_dict() if hasattr(result, "pvalues") else None,
        "scale": result.scale,
        "converged": result.mle_retvals.get("converged", None) if hasattr(result, "mle_retvals") else None
    }

    return output

@mcp.tool()
def test_stationarity(data, column, method='adf'):
    """
    Perform stationarity test (ADF or KPSS) on a time series column.

    Parameters:
        data: DataFrame or CSV path
        column: name of the column to test
        method: 'adf' or 'kpss'

    Returns:
        dict: Test statistic, p-value, and stationarity conclusion
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
        
    regression='c'
    series = data[column].dropna()

    if method == 'adf':
        result = adfuller(series)
        return {
            'method': 'ADF Test',
            'statistic': result[0],
            'p_value': result[1],
            'stationary': result[1] < 0.05
        }

    elif method == 'kpss':
        result = kpss(series, regression=regression, nlags='auto')
        return {
            'method': 'KPSS Test',
            'statistic': result[0],
            'p_value': result[1],
            'stationary': result[1] > 0.05
        }

    else:
        raise ValueError("method must be 'adf' or 'kpss'")

@mcp.tool()
def decompose_stl(data, column, period):
    """
    STL decomposition of a time series.

    Parameters:
        data: DataFrame or CSV path
        column: target time series column
        period: seasonal period (e.g., 12 for monthly data)

    Returns:
        dict: fitted model, and components' statistics
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    series = data[column].dropna()
    stl = STL(series, period=period)
    result = stl.fit()

    output = {
        "method": "STL Decomposition",
        "model": result,
        "trend_mean": result.trend.mean(),
        "trend_std": result.trend.std(),
        "seasonal_mean": result.seasonal.mean(),
        "resid_var": result.resid.var()
    }

    return output

@mcp.tool()
def auto_arima_modeling(data, column, seasonal=False, m=1, information_criterion='aic'):
    """
    Fit an ARIMA model with automatic order selection using auto_arima.

    Parameters:
        data: DataFrame or CSV path
        column: target column for ARIMA
        seasonal: whether to consider seasonality
        m: seasonal period (e.g., 12 for monthly)
        information_criterion: 'aic' or 'bic'

    Returns:
        dict: Fitted model details including order, coefficients, AIC, BIC, etc.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    series = data[column].dropna()

    model = auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        information_criterion=information_criterion,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False  
    )

    output = {
        "method": "auto_arima",
        "model": model,
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aic": model.aic(),
        "bic": model.bic(),
        "aicc": model.aicc(),
        "coefficients": dict(zip(model.arima_res_.params.index, model.arima_res_.params.values)),
        "pvalues": dict(zip(model.arima_res_.pvalues.index, model.arima_res_.pvalues.values)),
        "converged": model.arima_res_.mle_retvals.get("converged", None) if hasattr(model.arima_res_, "mle_retvals") else None
    }
    return output

@mcp.tool()
def kaplan_meier_plot(data, duration_col, event_col, group_col=None):
    """
    Plot Kaplan-Meier survival curve.

    Parameters:
        data (str or pd.DataFrame): Input data as CSV path or DataFrame.
        duration_col (str): Column name for survival time.
        event_col (str): Column name indicating event occurrence (1 = event, 0 = censored).
        group_col (str, optional): Column name for grouping (multiple curves).

    Returns:
        dict: Survival function data for plotting.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    kmf = KaplanMeierFitter()

    results = {}

    if group_col:
        for name, grouped_df in data.groupby(group_col):
            kmf.fit(grouped_df[duration_col], grouped_df[event_col], label=str(name))
            results[str(name)] = {
                "timeline": kmf.survival_function_.index.tolist(),
                "survival_prob": kmf.survival_function_[str(name)].tolist()
            }
    else:
        kmf.fit(data[duration_col], data[event_col], label="overall")
        results["overall"] = {
            "timeline": kmf.survival_function_.index.tolist(),
            "survival_prob": kmf.survival_function_["overall"].tolist()
        }

    return results

@mcp.tool()
def logrank_test_compare(data, duration_col, event_col, group_col):
    """
    Perform Log-rank test for survival difference between groups.

    Parameters:
        data (str or pd.DataFrame): Input data as CSV file path or DataFrame.
        duration_col (str): Column name for survival time.
        event_col (str): Column name indicating event occurrence (1 = event, 0 = censored).
        group_col (str): Column name used to define comparison groups.

    Returns:
        results: Summary of the log-rank test.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    groups = data[group_col].unique()
    if len(groups) == 2:
        g1, g2 = groups
        data1 = data[data[group_col] == g1]
        data2 = data[data[group_col] == g2]

        result = logrank_test(
            data1[duration_col], data2[duration_col],
            event_observed_A=data1[event_col],
            event_observed_B=data2[event_col]
        )
    else:
        result = multivariate_logrank_test(
            data[duration_col], data[group_col], data[event_col]
        )

    return result.summary

@mcp.tool()
def fit_cox_model(data, duration_col, event_col, covariates):
    """
    Fit a Cox proportional hazards regression model.

    Parameters:
        data (str or pd.DataFrame): Input data as CSV file path or DataFrame.
        duration_col (str): Column name for survival time.
        event_col (str): Column name indicating event occurrence (1 = event, 0 = censored).
        covariates (list of str): List of covariate (predictor) column names.

    Returns:
        dict: model, coefficients, hazard ratios, p-values, confidence intervals, and model info.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data[[duration_col, event_col] + covariates].dropna()


    dummy_col = df[covariates].select_dtypes(include=['object', 'category']).columns.tolist()
    if dummy_col:
        df = pd.get_dummies(df, columns=dummy_col, drop_first=True)

    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    summary = cph.summary.reset_index()
    if 'index' in summary.columns:
        summary = summary.rename(columns={'index': 'variable'})
    elif 'covariate' in summary.columns:
        summary = summary.rename(columns={'covariate': 'variable'})

    results = {
        "method": "CoxPH",
        "model": cph,
        "coefficients": summary.set_index("variable")["coef"].to_dict(),
        "hazard_ratios": summary.set_index("variable")["exp(coef)"].to_dict(),
        "pvalues": summary.set_index("variable")["p"].to_dict(),
        "ci95_lower": summary.set_index("variable")["exp(coef) lower 95%"].to_dict(),
        "ci95_upper": summary.set_index("variable")["exp(coef) upper 95%"].to_dict(),
        "model_info": {
            "concordance_index": cph.concordance_index_,
            "log_likelihood": cph.log_likelihood_,
            "AIC_partial": -2 * cph.log_likelihood_ + 2 * len(covariates)  # 类似 AIC
        }
    }

    return results

@mcp.tool()
def ab_ttest(data, group_col, value_col, group_A, group_B, alternative='two-sided'):
    """
    Two-sample t-test (Welch's) for A/B test.

    Parameters:
        data: DataFrame or CSV path
        group_col: column indicating group assignment (e.g., 'strategy')
        value_col: outcome metric to compare (e.g., 'spend', 'time')
        group_A: value in group_col representing group A
        group_B: value in group_col representing group B
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        dict: Test statistic, p-value, means, and conclusion
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    a = data[data[group_col] == group_A][value_col].dropna()
    b = data[data[group_col] == group_B][value_col].dropna()

    t_stat, p_value = ttest_ind(a, b, equal_var=False, alternative=alternative)

    interpretation = f"Mean of {group_A} = {a.mean():.3f}, {group_B} = {b.mean():.3f}\n"
    if p_value < 0.05:
        interpretation += f"✅ Significant difference (p = {p_value:.4f})"
    else:
        interpretation += f"❌ No significant difference (p = {p_value:.4f})"

    return {
        "method": "Two-sample t-test (Welch's)",
        "group_A_mean": a.mean(),
        "group_B_mean": b.mean(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "conclusion": interpretation
    }

@mcp.tool()
def bootstrap_abtest(
    data, 
    group_col, 
    value_col, 
    group_A, 
    group_B, 
    metric='mean', 
    n_bootstrap=5000, 
    ci=95,
    random_state=42
):
    """
    Bootstrap A/B test for difference in mean or median.

    Parameters:
        data: DataFrame or CSV path
        group_col: column that indicates A/B group
        value_col: column with outcome values
        group_A, group_B: group labels (e.g., 'A', 'B')
        metric: 'mean' or 'median'
        ci: confidence level (e.g., 95)

    Returns:
        dict: Observed difference, CI bounds, significance, and conclusion
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    np.random.seed(random_state)

    a_vals = data[data[group_col] == group_A][value_col].dropna().values
    b_vals = data[data[group_col] == group_B][value_col].dropna().values

    boot_diffs = []

    for _ in range(n_bootstrap):
        a_sample = np.random.choice(a_vals, size=len(a_vals), replace=True)
        b_sample = np.random.choice(b_vals, size=len(b_vals), replace=True)

        if metric == 'mean':
            diff = np.mean(b_sample) - np.mean(a_sample)
        elif metric == 'median':
            diff = np.median(b_sample) - np.median(a_sample)
        else:
            raise ValueError("metric must be 'mean' or 'median'")

        boot_diffs.append(diff)

    lower = np.percentile(boot_diffs, (100 - ci) / 2)
    upper = np.percentile(boot_diffs, 100 - (100 - ci) / 2)
    observed = (np.mean(b_vals) - np.mean(a_vals)) if metric == 'mean' else (np.median(b_vals) - np.median(a_vals))

    significant = not (lower <= 0 <= upper)
    interpretation = (
        f"✅ Significant difference: CI = ({lower:.4f}, {upper:.4f}) excludes 0"
        if significant else
        f"❌ No significant difference: CI = ({lower:.4f}, {upper:.4f}) includes 0"
    )

    return {
        "method": f"Bootstrap A/B Test ({metric})",
        "observed_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": significant,
        "conclusion": interpretation
    }

@mcp.tool()
def ab_power_analysis(test_type='proportion',
                      baseline=0.02,
                      effect=0.005,
                      std_dev=None,
                      alpha=0.05,
                      power=0.8,
                      alternative='larger'):
    """
    Calculate required sample size per group for A/B test power analysis.

    Parameters:
        test_type: 'proportion' or 'mean'
        baseline: baseline value (CTR or mean)
        effect: minimum detectable effect (absolute increase)
        std_dev: standard deviation (required for test_type='mean')
        alpha: significance level (default 0.05)
        power: statistical power (default 0.8)
        alternative: 'two-sided', 'larger', or 'smaller'

    Returns:
        Required sample size per group (rounded up)
    """
    if test_type == 'proportion':
        # H0: p1 = baseline, H1: p2 = baseline + effect
        p1 = baseline
        p2 = baseline + effect
        effect_size = proportion_effectsize(p1, p2)

        power_analysis = NormalIndPower()
        n = power_analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                       power=power, alternative=alternative)

    elif test_type == 'mean':
        if std_dev is None:
            raise ValueError("std_dev must be provided for mean test.")
        effect_size = effect / std_dev  # Cohen's d
        power_analysis = TTestIndPower()
        n = power_analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                       power=power, alternative=alternative)
    else:
        raise ValueError("test_type must be 'proportion' or 'mean'")

    return math.ceil(n)

@mcp.tool()
def advanced_regression(data, target, method='lasso', test_size=0.2, random_state=42):
    """
    Perform high-dimensional regression using Lasso, Ridge, or ElasticNet with cross-validation.

    Parameters:
        data: DataFrame or CSV path
        target: target column name
        method: 'lasso', 'ridge', or 'elasticnet'
        test_size: test set ratio
        random_state: random seed

    Returns:
        dict with:
            - method
            - model
            - best_alpha
            - coefficients
            - r2
            - rmse
            - y_test
            - y_pred
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    X = data.drop(columns=[target])
    y = data[target]

    # Drop non-numeric columns (or handle encoding outside)
    X = X.select_dtypes(include=['float64', 'int64'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if method == 'lasso':
        model = LassoCV(cv=5, random_state=random_state)
    elif method == 'ridge':
        model = RidgeCV(cv=5)
    elif method == 'elasticnet':
        model = ElasticNetCV(cv=5, random_state=random_state)
    else:
        raise ValueError("Method must be 'lasso', 'ridge', or 'elasticnet'")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        "method": method,
        "model": model,
        "best_alpha": getattr(model, 'alpha_', None),
        "coefficients": dict(zip(X.columns, model.coef_)),
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred),
        "model": model,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }

@mcp.tool()
def sparse_pca_analysis(data, n_components=5, alpha=1, max_iter=1000, top_features=5):
    """
    Perform Sparse PCA to extract interpretable components.

    Parameters:
        data: DataFrame or CSV path (only numerical columns will be used)
        n_components: number of components to extract

    Returns:
        dict with:
            - component_matrix: feature loadings
            - reduced_data: transformed low-dim data
            - top_features_per_component
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    data = data.select_dtypes(include=['float64', 'int64']).dropna()
    feature_names = data.columns

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Sparse PCA
    spca = SparsePCA(n_components=n_components, alpha=alpha, max_iter=max_iter, random_state=42)
    X_reduced = spca.fit_transform(X_scaled)
    components = spca.components_

    comp_df = pd.DataFrame(components, columns=feature_names)

    top_features_dict = {}
    for i in range(n_components):
        comp = comp_df.iloc[i]
        top = comp.abs().sort_values(ascending=False).head(top_features)
        top_features_dict[f"Component {i+1}"] = top.index.tolist()

    return {
        "component_matrix": comp_df,
        "reduced_data": pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)]),
        "top_features_per_component": top_features_dict
    }

@mcp.tool()
def bayesian_inference(
    model: Literal["binomial", "normal"],
    data: dict,
    prior: dict,
    ci: float = 0.95
) -> dict:
    """
    Bayesian posterior inference for binomial or normal models.

    Parameters:
        model: "binomial" or "normal"
        data:
            if binomial: {"successes": int, "trials": int}
            if normal: {"mean": float, "n": int, "std": float}  # sample std
        prior:
            if binomial: {"alpha": float, "beta": float}
            if normal: {"mu0": float, "sigma0": float}
        ci: credible interval level (default 0.95)

    Returns:
        dict: posterior mean, credible interval, and posterior distribution description
    """
    if model == "binomial":
        s = data["successes"]
        n = data["trials"]
        a0 = prior.get("alpha", 1)
        b0 = prior.get("beta", 1)

        a_post = a0 + s
        b_post = b0 + (n - s)

        lower, upper = stats.beta.ppf([(1 - ci) / 2, (1 + ci) / 2], a_post, b_post)
        return {
            "posterior_mean": a_post / (a_post + b_post),
            "credible_interval": (lower, upper),
            "posterior_dist": f"Beta({a_post:.1f}, {b_post:.1f})"
        }

    elif model == "normal":
        x̄ = data["mean"]
        n = data["n"]
        s = data["std"]
        mu0 = prior["mu0"]
        sigma0 = prior["sigma0"]

        # Known population variance assumed = s^2
        sigma_sq = s**2
        sigma0_sq = sigma0**2

        posterior_mean = (sigma0_sq * x̄ + sigma_sq * mu0 / n) / (sigma0_sq + sigma_sq / n)
        posterior_std = np.sqrt(1 / (1 / sigma0_sq + n / sigma_sq))

        lower, upper = stats.norm.ppf([(1 - ci) / 2, (1 + ci) / 2], loc=posterior_mean, scale=posterior_std)
        return {
            "posterior_mean": posterior_mean,
            "credible_interval": (lower, upper),
            "posterior_dist": f"N({posterior_mean:.1f}, {posterior_std:.1f}²)"
        }

    else:
        raise ValueError("model must be 'binomial' or 'normal'")

@mcp.tool()
def bayesian_linear_regression(data, target, predictor, credible_interval=0.95):
    """
    Perform Bayesian linear regression using BayesianRidge.

    Parameters:
        data: DataFrame or CSV file path
        target: name of target variable (e.g. 'sales')
        predictor: name of predictor variable (e.g. 'ad_spend')
        credible_interval: desired credible interval (default 0.95)

    Returns:
       dict with model, posterior mean, std, credible interval, and P(coef < 0)
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data[[target, predictor]].dropna()

    X = df[[predictor]].values
    y = df[target].values

    # Optional: standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = BayesianRidge()
    model.fit(X_scaled, y_scaled)

    coef_mean = model.coef_[0]
    coef_std = np.sqrt(model.sigma_[0, 0])  # variance of the coefficient

    alpha = 1 - credible_interval
    ci_low, ci_high = norm.ppf([alpha / 2, 1 - alpha / 2], loc=coef_mean, scale=coef_std)

    # Probability that coefficient < 0
    prob_negative = norm.cdf(0, loc=coef_mean, scale=coef_std)

    return {
        "method": "Bayesian Linear Regression",
        "model": model,
        "posterior_mean": coef_mean,
        "posterior_std": coef_std,
        "credible_interval": (ci_low, ci_high),
        "P(coef < 0)": prob_negative
    }

@mcp.tool()
def fit_hierarchical_model(data, group_col, outcome_col, predictor_col=None, model_type="normal", draws=200, tune=200):
    """
    Fit a hierarchical Bayesian model using PyMC.

    Parameters:
        data: DataFrame or CSV file path
        group_col: column for group ID (e.g. individual, town, school)
        outcome_col: column with observed values (e.g. weight, CTR, score)
        predictor_col: optional column for predictor (e.g. year, impressions)
        model_type: 'normal' for continuous outcomes, 'binomial' for clicks/impressions
        draws: number of MCMC samples
        tune: tuning steps

    Returns:
        dict with model, trace, and summary statistics
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    groups = df[group_col].astype("category").cat.codes.values
    n_groups = df[group_col].nunique()

    with pm.Model() as model:
        if model_type == "normal":
            mu_a = pm.Normal("mu_a", mu=0, sigma=10)  # group intercept mean
            sigma_a = pm.HalfNormal("sigma_a", sigma=10)  # group intercept std

            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_groups)  # group-specific intercepts

            if predictor_col:
                x = df[predictor_col].values
                b = pm.Normal("b", mu=0, sigma=10)
                mu = a[groups] + b * x
            else:
                mu = a[groups]

            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal("y", mu=mu, sigma=sigma, observed=df[outcome_col].values)

        elif model_type == "binomial":
            clicks = df[outcome_col].values
            impressions = df[predictor_col].values  # predictor = impressions

            mu_logit = pm.Normal("mu_logit", 0, 1)
            sigma_logit = pm.HalfNormal("sigma_logit", 1)

            theta_logit = pm.Normal("theta_logit", mu=mu_logit, sigma=sigma_logit, shape=n_groups)
            theta = pm.Deterministic("theta", pm.math.sigmoid(theta_logit[groups]))

            y = pm.Binomial("y", n=impressions, p=theta, observed=clicks)

        else:
            raise ValueError("model_type must be 'normal' or 'binomial'")

        trace = pm.sample(draws=draws, tune=tune, target_accept=0.9, progressbar=False, return_inferencedata=True)
    summary = az.summary(trace, round_to=2)

    results = {
        "model": model,
        "trace": trace,
        "summary": summary
    }
    return results

@mcp.tool()
def estimate_ATT_with_psm(data, treatment, outcome, covariates, treatment_value=1, caliper=0.05, standardize=True):
    """
    Estimate ATT using Propensity Score Matching (1:1 Nearest Neighbor, no replacement).

    Parameters:
        data: DataFrame or CSV file path
        treatment: binary treatment column name (e.g., 'Private')
        outcome: outcome column name (e.g., 'PhD', 'Grad.Rate')
        covariates: list of covariate column names to match on
        treatment_value: value of treatment indicating treated group (default 1)
        caliper: maximum distance for matching
        standardize: whether to scale covariates before propensity score estimation

    Returns:
        dict with ATT estimate, matched sample size, and matched DataFrame
    """
    data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

    df = data.copy().dropna(subset=[treatment, outcome] + covariates)

    # 1. Create binary treatment indicator
    df['T'] = (df[treatment] == treatment_value).astype(int)

    # 2. Estimate propensity scores
    X = df[covariates].values
    if standardize:
        X = StandardScaler().fit_transform(X)
    y = df['T'].values
    model = LogisticRegression()
    model.fit(X, y)
    df['pscore'] = model.predict_proba(X)[:, 1]

    # 3. Separate treated and control
    treated = df[df['T'] == 1].copy()
    control = df[df['T'] == 0].copy()

    # 4. Nearest Neighbor Matching (1:1 within caliper)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['pscore']])
    distances, indices = nn.kneighbors(treated[['pscore']])

    matched_pairs = []
    used_indices = set()
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper and idx not in used_indices:
            matched_pairs.append((treated.index[i], control.index[idx]))
            used_indices.add(idx)

    matched_treated = df.loc[[i for i, _ in matched_pairs]]
    matched_control = df.loc[[j for _, j in matched_pairs]]
    matched_df = pd.concat([
        matched_treated.assign(group='treated'),
        matched_control.assign(group='control')
    ])

    # 5. Estimate ATT
    att = matched_treated[outcome].mean() - matched_control[outcome].mean()

    return {
        "ATT": att,
        "n_matched": len(matched_pairs),
        "matched_data": matched_df.reset_index(drop=True)
    }

@mcp.tool()
def estimate_did_effect(
    data,
    outcome,
    treat_col,
    time_col,
    post_year,
    placebo_year=None,
    cluster=None
):
    """
    Estimate the Difference-in-Differences (DID) effect using panel data.

    Parameters:
        data: DataFrame or CSV file path
        outcome: name of outcome variable (e.g., 'accidents')
        treat_col: binary treatment group column (1 = treatment group)
        time_col: column indicating year (e.g., 2017–2022)
        post_year: year the real policy was enacted
        placebo_year: optional placebo year (e.g., 2019)
        cluster: optional column name to cluster standard errors (e.g., state ID)

    Returns:
        dict with DID coefficient, p-value, confidence interval, model summary, and placebo results
    """
    data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    df = data.copy()

    # 1. Define post indicator
    df['post'] = (df[time_col] >= post_year).astype(int)

    # 2. Treatment × Post interaction term
    df['did'] = df['post'] * df[treat_col]

    # 3. Define formula
    formula = f"{outcome} ~ {treat_col} + post + did"

    # 4. Fit model
    if cluster:
        model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[cluster]})
    else:
        model = smf.ols(formula, data=df).fit()

    result = {
        "DID_coefficient": model.params['did'],
        "p_value": model.pvalues['did'],
        "conf_int": model.conf_int().loc['did'].tolist(),
        "model": model,
        "placebo": None
    }

    # 5. Optionally run placebo DID
    if placebo_year:
        df['post_placebo'] = (df[time_col] >= placebo_year).astype(int)
        df['did_placebo'] = df['post_placebo'] * df[treat_col]
        placebo_formula = f"{outcome} ~ {treat_col} + post_placebo + did_placebo"
        if cluster:
            placebo_model = smf.ols(placebo_formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[cluster]})
        else:
            placebo_model = smf.ols(placebo_formula, data=df).fit()
        result["placebo"] = {
            "Placebo_DID": placebo_model.params['did_placebo'],
            "p_value": placebo_model.pvalues['did_placebo'],
            "conf_int": placebo_model.conf_int().loc['did_placebo'].tolist(),
            "model": placebo_model
        }

    return result

@mcp.tool()
def synthetic_control(
    df,
    time_col,
    unit_col,
    outcome_col,
    treated_unit,
    intervention_time,
    control_units=None,
    weights=None,
    plot=True
):
    """
    Synthetic Control Estimation Function

    Parameters:
        df: DataFrame or CSV file path
        time_col: column name for time (e.g. 'year')
        unit_col: column name for unit ID (e.g. 'city_id' or 'state')
        outcome_col: column name for outcome variable (e.g. 'unemployment')
        treated_unit: the ID of the treated unit (e.g. 1)
        intervention_time: year when treatment begins (e.g. 2015)
        control_units: optional list of control unit IDs
        weights: optional list of weights for control units (must sum to 1)

    Returns:
        dict with time, treated, synthetic, treatment_effect, and weights
    """

    df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
    df = df.copy()
    df = df[[time_col, unit_col, outcome_col]].dropna()

    # Subset data
    treated_df = df[df[unit_col] == treated_unit]
    all_units = df[unit_col].unique()
    if control_units is None:
        control_units = [u for u in all_units if u != treated_unit]

    control_df = df[df[unit_col].isin(control_units)]

    time_points = sorted(df[time_col].unique())
    outcome_matrix = []

    for unit in control_units:
        unit_data = control_df[control_df[unit_col] == unit].sort_values(time_col)
        outcome_matrix.append(unit_data[outcome_col].values)

    Y_controls = np.array(outcome_matrix).T  # shape: time × units
    Y_treated = treated_df.sort_values(time_col)[outcome_col].values

    if weights is None:
        # Train weights by minimizing squared error in pre-intervention period
        pre_idx = df[time_col].unique() < intervention_time
        Y0 = Y_controls[pre_idx, :]  # T0 × N
        Y1 = Y_treated[pre_idx]      # T0

        # Solve min ||Y1 - Y0 w||^2  s.t. sum(w)=1, w≥0
        from cvxpy import Variable, Minimize, Problem, sum, quad_form
        N = Y0.shape[1]
        w = Variable(N)
        objective = Minimize(quad_form(Y1 - Y0 @ w, np.eye(len(Y1))))
        constraints = [sum(w) == 1, w >= 0]
        prob = Problem(objective, constraints)
        prob.solve()
        weights = w.value

    weights = np.array(weights)
    Y_synthetic = Y_controls @ weights
    treatment_effect = Y_treated - Y_synthetic

    return {
        "time": time_points,
        "treated": Y_treated,
        "synthetic": Y_synthetic,
        "treatment_effect": treatment_effect,
        "weights": weights
    }

@mcp.tool()
def fdr_control_df(
    data,
    pval_col='pval',
    alpha=0.05,
    method='bh'
):
    """
    Perform FDR control (BH/BY) on a DataFrame or CSV file containing p-values.

    Parameters:
        data: pd.DataFrame or str (CSV path)
        pval_col: name of the column containing p-values
        alpha: significance level for FDR control (default 0.05)
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns:
        pd.DataFrame with additional columns 'qval' and 'significant'
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be a DataFrame or a CSV file path.")

    # Validate p-values
    if pval_col not in df.columns:
        raise ValueError(f"Column '{pval_col}' not found in data.")
    pvals = df[pval_col].values
    n = len(pvals)

    # Sort p-values
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH or BY adjustment
    if method == 'bh':
        scale = n / np.arange(1, n+1)
    elif method == 'by':
        c_m = np.sum(1 / np.arange(1, n+1))
        scale = n * c_m / np.arange(1, n+1)
    else:
        raise ValueError("method must be 'bh' or 'by'")

    # Compute q-values
    qvals_sorted = np.minimum.accumulate((scale * sorted_pvals)[::-1])[::-1]
    qvals = np.empty(n)
    qvals[sorted_idx] = qvals_sorted

    # Add result columns
    df['qval'] = qvals
    df['significant'] = df['qval'] <= alpha

    return df

@mcp.tool()
def fwer_control_df(
    data,
    pval_col='pval',
    alpha=0.05,
    method='bonferroni'
):
    """
    Perform FWER control (Bonferroni or Holm) on p-values in a DataFrame or CSV.

    Parameters:
        data: pd.DataFrame or str (CSV file path)
        pval_col: name of the column with raw p-values
        alpha: overall FWER threshold (e.g., 0.05)
        method: 'bonferroni' or 'holm'

    Returns:
        DataFrame with original data + adjusted p-values and significance flag
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be a DataFrame or a CSV file path.")

    if pval_col not in df.columns:
        raise ValueError(f"Column '{pval_col}' not found in data.")

    pvals = df[pval_col].values
    n = len(pvals)

    if method == 'bonferroni':
        adj_pvals = np.minimum(pvals * n, 1.0)
    elif method == 'holm':
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]
        adj_sorted = np.empty(n)
        for i, p in enumerate(sorted_pvals):
            adj_sorted[i] = min((n - i) * p, 1.0)
        adj_sorted = np.maximum.accumulate(adj_sorted)  # 保持单调
        adj_pvals = np.empty(n)
        adj_pvals[sorted_idx] = adj_sorted
    else:
        raise ValueError("method must be 'bonferroni' or 'holm'.")

    df['adj_pval'] = adj_pvals
    df['significant'] = df['adj_pval'] <= alpha

    return df

@mcp.tool()
def show_csv_info_en(data: str):
    """
    Display basic information about a dataset.

    Parameters:
        data (str): Path to the CSV file.

    Returns:
        str: Summary of the dataset including name, first 5 rows, and first 50 column names.
    """

    data1 = pd.read_csv(data)

    output = f"""
    - Dataset name
    {os.path.basename(data)}
    - Dataset first 5 rows:
    {data1.head().to_string(index=False)}
    - Dataset first 50 columns' names:
    {list(data1.columns)[:50]}
    """
    return output


@mcp.tool()
def read_file(file_path: str):
    """
    Read the content of a file based on its extension.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        The content of the file.
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    if file_path.endswith('md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


if __name__ == "__main__":
    mcp.run(transport="stdio")