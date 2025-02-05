import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_tukey



def set_plotting_params(f_size=15,
                        t_size=18,
                        t_dir='in',
                        major=5.0,
                        minor=3.0,
                        min_size=3.0,
                        maj_size=5.0,
                        l_width=0.7,
                        l_handle=2.0):
    """Set Matplotlib parameters.

    Args:
        f_size: int
            The font size.
        t_size: int
            The legend font size.
        t_dir: string,
            The direction of y and y ticks.
        major: float
            The x and y ticks major.
        minor: float
            The x and y ticks minor.
        min_size: float
            Axes minor size.
        max_size: float
            Axes major size.
        l_width: float
            The axes line width.
        l_handle: float
            The legend handle length.

    Return:
        None
    """
    plt.style.use('bmh')
    plt.rcParams['font.size'] = f_size
    plt.rcParams['legend.fontsize'] = t_size
    plt.rcParams['xtick.direction'] = t_dir
    plt.rcParams['ytick.direction'] = t_dir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = min_size
    plt.rcParams['ytick.minor.size'] = maj_size
    plt.rcParams['axes.linewidth'] = l_width
    plt.rcParams['legend.handlelength'] = l_handle


def load_data(ds_path, output_var, drop_cols=None):
    """Loads a dataset with pandas from a given path.

    Args:
        ds_path: string
            The path of the dataset.
        output_var: string
            The output variable of in the dataset. In our case the cognitive variable.
        keep_cols: array, default None
            The columns to keep when loading the dataset.

    Returns:
        X: array(n_sample, n_features)
            The train independent variables.
        y: array(n_sample,)
            The output/dependent variable.
    """

    df = pd.read_csv(ds_path)
    print(list(df))
    X, y = df.drop([output_var], axis=1), df[output_var]
    if drop_cols:
        X = X.drop(drop_cols, axis=1)
    X = X.rename(
        {
            'VIGOURIOUS ACT': 'VIGOURIOUS ACTIVITIES',
            'MODERATE ACT': 'MODERATE ACTIVITIES',
            'MILD ACT': 'MILD ACTIVITIES',
            'ONGOING HOUSING PREOBLEMS': 'ONGOING HOUSING PROBLEMS',
        },
        axis=1)
    X = make_col_names_title_format(X)
    y = y.astype(int)
    return X, y


def make_col_names_title_format(df):
    """Rename the columns of a dataframe to a title format.

    For instance from 'OFTEN DO HOBBY' to 'Often Do Hobby'.

    Args:
        df: array(n_sample, n_features)
            The dataframe to rename the columns.
    Returns:
        df: array(n_sample, n_features)
            The dataframe with columns renamed.
    """
    columns = pd.Series(list(df))
    columns = columns.str.title()
    df.columns = columns
    return df


def get_linear_eq(x, w0, w1, beta0):
    """Returns the linear equation (y= beta0 + wx) used in our work.

    Args:
        x: array(n_xticks)
            Binary variable to plot. We consider a range of values between
            -1 and 1 where x < 0 correspond to the binary category 0 and x > 0
            the binary category 1 (performing a given lifestyle activity).
        w0: float
            The weight (logit) associated the category 0.
        w1: w0: float
            The weight (logit) associated the category 1.
        beta0: float
            The intercept associated to the output category y.
    Returns:
        y: array(n_xticks)
            The array corresponding to the linear equation. If x < 0,
            y = beta0 + w0x, if x > 0, y = beta0 + w1x otherwise y = beta0.
    """
    if x < 0:
        y = beta0 + w0
    elif x > 0:
        y = beta0 + w1
    else:
        y = beta0
    return y


def get_errors(explanation, class_label, var_type="Categorical", x_lim=None):
    """Returns the errors (standard deviations) associated to each variable
    categories and class label.

    For binary variables, we plot the errors corresponding to the category 0
    when x < 0 and inversely for category 1. By default, the EBM model
    discretizes continuous variables and associates a weight and errors to
    each category.

    Args:
        explanation: dict
            The explanation data corresponding to the class label class_label.
        class_label: int,
            The class label to get the errors.
        var_type: string, default "Categorical"
            The type of variable to return the errors. Possible values are
            "Categorical" and "Continuous".
        x_lim: array(n_x_lim)
            The x lim labels for binary variables. Use to to create same
            errors for category 0 when x < 0 and conversly for category 1.
    Returns:
        errors: tuple
            The errors corresponding to each variable category.
    """
    if var_type == "Continuous":
        up_err = explanation["upper_bounds"][:, class_label]
        low_err = explanation["lower_bounds"][:, class_label]
        errors = (low_err, up_err)
    else:
        up_err = explanation["upper_bounds"][:, class_label]
        low_err = explanation["lower_bounds"][:, class_label]
        up_err_cat_0 = up_err[0]
        up_err_cat_1 = up_err[1]
        low_err_cat_0 = low_err[0]
        low_err_cat_1 = low_err[1]

        len_xlim = len(x_lim)
        new_up_err = np.zeros((len_xlim, ))
        new_up_err[x_lim < 0] = up_err_cat_0
        new_up_err[x_lim > 0] = up_err_cat_1
        new_up_err[x_lim == 0] = 0

        new_low_err = np.zeros((len_xlim, ))
        new_low_err[x_lim < 0] = low_err_cat_0
        new_low_err[x_lim > 0] = low_err_cat_1
        new_low_err[x_lim == 0] = 0
        errors = (new_low_err, new_up_err)
    return errors


def save_pvalues(p_values, path_save, thresold=0.05):
    """Formats p_values when saving to a latex format.

    Args:
        p_values: array(n_features, n_cogn_cat)
            Dataframe containing p-values.
        path_save: string
            The path to save the p-values to latex file.
        threshold: float, default 0.05
            The threshold to use to format p-values.
    Returns:
        None
    """
    def format_bold(x):
        if x < thresold:
            return "\\textbf{%s}" % x
        else:
            return x

    df = p_values.copy()
    for col in list(p_values):
        df[col] = p_values[col].apply(lambda x: format_bold(x))
    df.to_latex(path_save + ".tex", escape=False)

    # Formatting significant bins to Markdown bold.
    # df_md = df.replace("\\textbf{", "**")
    # df_md = df_md.replace("}", "**")
    # df_md.to_markdown(path_save + ".md", index=True, tablefmt="grid")

def stat_comparisons(scores_kf, file_name):
    ebm = scores_kf["EBM"]
    xgb = scores_kf["XGB"]
    lr = scores_kf["LR"]
    svc = scores_kf["SVC"]
    mlp = scores_kf["MLP"]
    rf = scores_kf["RF"]

    df_posthoc = pd.DataFrame()
    n = len(ebm)
    df_posthoc["scores"] = np.r_[ebm, xgb, lr, svc, mlp, rf]
    df_posthoc["models"] = ["EBM"]*n + ["XGB"]*n + ["LR"]*n + ["SVC"]*n + ["MLP"]*n + ["RF"]*n
    print(kruskal(ebm, xgb, lr, svc, mlp, rf))
    tukey_df = posthoc_tukey(df_posthoc, val_col="scores", group_col="models").round(3)
    print(tukey_df)
    tukey_df.to_latex(file_name + "_tukey.tex", escape=False)
    tukey_df.to_csv(file_name + "_tukey.csv", header=True, index=False)

