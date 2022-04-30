import os

import matplotlib.pyplot as plt
import numpy as np
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
        ax.set_ylabel("Logit" if idx in (0, 4, 8) else "")
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
