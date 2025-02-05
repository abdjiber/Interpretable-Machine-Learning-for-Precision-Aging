import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

import src.utils as utils
from src.config import PLOTS_COLORS, PLOTS_LINESTYLES


def plot_linear_eqs(explainer, save_dir, fig_name, type_features="Important"):
    """Plots linear equations between the dependent and independent variables.

    Args:
        explainer: Explain
            The instance of Explain class.
        save_dir: string
            The path of results saving.
        fig_name: string
            The name of the figure file to save.
        type_features: string, default "Important"
            The type of features to plot. Possible values are "Important" and "All".
    Returns:
        None
    """
    original_features = explainer.features
    top_features = explainer.top_imp_features.index
    global_exp = explainer.global_exp
    intercepts = explainer.intercepts
    classes = explainer.class_names

    df_scores = pd.DataFrame()
    utils.set_plotting_params()  # Setting the plotting parameters
    if type_features == "Important":
        n_cols, n_lines = (4, 3)
        figsize = (18, 12)
        features = top_features
    else:
        n_cols, n_lines = (4, 9)
        figsize = (25, 30)
        features = original_features

    fig = plt.figure(figsize=figsize)
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(n_lines, n_cols, idx + 1)
        explanation = global_exp.data(original_features.index(feature))
        scores = explanation["scores"]
        var_categories = explanation["names"]
        classes = explanation["meta"]["label_names"]
        if feature in ("Age", "Education"):
            for idx_cl, _ in enumerate(classes):
                (low_err, up_err) = utils.get_errors(explanation,
                                                     idx_cl,
                                                     var_type="Continuous")
                color = PLOTS_COLORS[idx_cl] if len(classes) <= 5 else None
                linestyle = PLOTS_LINESTYLES[idx_cl] if len(
                    classes) <= 5 else None
                ax.errorbar(var_categories[:-1],
                            y=scores[:, idx_cl],
                            yerr=np.array([low_err, up_err]),
                            color=color,
                            label=str(classes[idx_cl]),
                            linestyle=linestyle,
                            linewidth=7,
                            elinewidth=0.7)
        else:
            linear_eq = np.vectorize(utils.get_linear_eq)
            for idx_cl, _ in enumerate(classes):
                w1 = scores[1][idx_cl]
                w0 = scores[0][idx_cl]
                b = intercepts[idx_cl]
                x = np.arange(-1, 1.25, 0.25)
                (low_err, up_err) = utils.get_errors(explanation,
                                                     idx_cl,
                                                     var_type="Categorical",
                                                     x_lim=x)

                y = linear_eq(x, w0, w1, b)
                color = PLOTS_COLORS[idx_cl] if len(classes) <= 5 else None
                linestyle = PLOTS_LINESTYLES[idx_cl] if len(
                    classes) <= 5 else None
                ax.errorbar(x,
                            y,
                            yerr=np.array([low_err, up_err]),
                            color=color,
                            label=str(classes[idx_cl]),
                            linestyle=linestyle,
                            linewidth=6,
                            elinewidth=0.7)
        ax.set_xlabel(feature)
        ax.set_ylabel("Logit" if idx in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40) else "")
        if idx in (3, 7, 11, 15, 19, 23, 27, 31, 35):
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    fig.tight_layout()
    file_name = os.path.join(save_dir, fig_name)
    fig.savefig(file_name)


def plot_zscores(z_scores,
                 y_label="z-scores",
                 x_label="Cognitive Categories",
                 figsize=(10, 8),
                 file_name="z-scores.png"):
    """Plots the z-scores obtained from the statistical significance tests.

    Args:
        z_scores: array(n_bins, n_cogn_categories)
                Array of z-scores values where n_bins is the total number of
                bins of selected variables and n_cogn_categories the number of
                cognitive categories.
        y_label: string, default "z_scores"
            The y-axis label.
        x_label: string, default "Cognitive Categories"
            The x-axis label.
        figsize: tuple(int, int), default (10, 8)
        file_name: string, defaul "z-scores.png"
            The file name of the figure to save.

    """
    plt.rcParams['legend.fontsize'] = 11
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = z_scores.T.plot.bar(colormap=cm.Spectral,
                             width=1,
                             figsize=figsize,
                             ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(y=1.96, linestyle="--")
    ax.axhline(y=-1.96, linestyle="--")
    ax.legend(bbox_to_anchor=(0, 1, 1, 0),
              loc="lower left",
              mode="expand",
              ncol=3)
    fig.savefig(file_name + ".png", bbox_inches='tight', dpi=100)


def plot_effects(df, file_name):
    # Set up the plot
    df_lifestyles = df[~df["Category"].isin(["Age", "Education"])]
    factors = df_lifestyles.Factors.unique()
    for factor in factors:
        df_factor = df_lifestyles[df_lifestyles["Factors"] == factor]
        fig = plot_factors(df_factor, factor, is_age_edu=False)
        fig.savefig(file_name + f"_factor_{factor}.png", dpi=100)
    
    categories = df["Category"].unique()
    if "Age" in categories:
        df_age =  df[df.Category == "Age"]
        df_age.iloc[:, :] = df_age.apply(pd.to_numeric, errors='coerce')
        fig = plot_factors(df_age, factor="Age", is_age_edu=True)
        fig.savefig(file_name + f"_age.png", dpi=100)

    if "Education" in categories:
        df_edu =  df[df.Category == "Education"]
        df_edu.iloc[:, :] = df_edu.apply(pd.to_numeric, errors='coerce')
        fig = plot_factors(df_edu, factor="Education", is_age_edu=True)
        fig.savefig(file_name + f"_education.png", dpi=100)
    

def plot_factors(df_factor, factor=None, is_age_edu=False):
    fig, ax = plt.subplots(figsize=(14, 12))
    # Define colors for each group
    # colors = ['red', 'blue', 'green']
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'black', 'purple', 'orange']
    groups = [col for col in df_factor.columns if "Score" in col] # ["Score G1", "Score G2", "Score G3"]
    err_lows = [col for col in df_factor.columns if "Low" in col] # ["Err Low G1", "Err Low G2", "Err Low G3"]
    err_highs = [col for col in df_factor.columns if "High" in col] # ["Err High G1", "Err High G2", "Err High G3"]
    new_labels = ["Low", "Central", "High"] if len(groups) == 3 else [f"Cogn {i}" for i in range(1, len(groups) + 1)]
    _colors = colors[:len(groups)]
    # Loop through each group and plot
    for i, (group, err_low, err_high, color, label) in enumerate(zip(groups, err_lows, err_highs, _colors, new_labels)):
        err = [df_factor[group] - df_factor[err_low], df_factor[err_high] - df_factor[group]]
        if not is_age_edu:
            x = df_factor[group]
            y = range(len(df_factor))
            fmt = "o"
        else:
            x = df_factor["Factors"]
            y = df_factor[group]
            fmt="-o"
       
        ax.errorbar(
            x, y,
            xerr=err if not is_age_edu else None,
            yerr=err if is_age_edu else None,
            fmt=fmt, color=color, label=label,
            capsize=3, elinewidth=1, markeredgewidth=1
        )

    if not is_age_edu:
    # Add a vertical reference line at x = 0
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="Zero Reference")

        # Set x-axis limits to range from -0.25 to 0.25
        ax.set_xlim(-0.3, 0.3)

        # Formatting
        ax.set_yticks(range(len(df_factor)))
        ax.set_yticklabels(df_factor["Category"])
        ax.set_xlabel("Score")
        ax.set_title(f"Effect Sizes of Activities Across Groups (Factor = {factor})")
        ax.invert_yaxis()  # Invert y-axis for readability
    else:
        # ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel(f"{factor} (Years)")
        ax.set_ylabel("Score")
        ax.set_title(f"Score Trends by {factor} with Confidence Intervals")

    # Show the updated plot
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
