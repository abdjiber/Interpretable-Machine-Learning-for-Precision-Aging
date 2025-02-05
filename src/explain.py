import numpy as np
import pandas as pd
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier
from scipy import stats

import src.plotting as plotting
import src.utils as utils
from src.type_experiments import TypeExp
from src.config import RANDOM_SEED

import interpret
print(interpret.__version__)

class Explain:
    """Class used to provide explanations of the EBM model.

    This class is used to provide feature importance, plot a linear\
    relationship between covariates and the dependent variable, and\
    finally to compute z-scores and associated p-values of considered\
    variable bins.

    Args:
        X: array(n_sample, n_features)
            The training dataset.
        y: array(n_sample)
            The output variable.
        type_exp: TypeExp
            The type of experiments to run. Possible values are\
            TypeExp.WITH_AGE_EDUCATION and TypeExp.WITHOUT_AGE_EDUCATION.
    """
    def __init__(self, X, y, type_exp):
        """Inits the Explain class."""
        self.X = X
        self.y = y
        self.exp_type = type_exp
        self.features = list(X)
        self.class_names = None
        self.intercepts = None
        self.standard_devs = None
        self.ebm = None
        self.global_exp = None
        self.feature_imp = None
        self.top_imp_features = None
        self.z_scores = None
        self.p_values = None

    def fit(self):
        """Fits an EBM model"""
        np.random.seed(RANDOM_SEED)
        ebm = ExplainableBoostingClassifier()
        ebm.fit(self.X, self.y)
        ebm_global = ebm.explain_global()
        self.ebm = ebm
        self.class_names = ebm.classes_
        self.feature_imp = ebm.feature_importances_
        self.standard_devs = ebm.term_standard_deviations_
        self.intercepts = ebm.intercept_
        self.global_exp = ebm_global

    def get_factor_effects(self):
        elements = ["Score", "Err Low", "Err High"]
        categories_factors = ["Category", "Factors"]

        ebm_global = self.global_exp
        classes = ebm_global.data(0)["meta"]["label_names"]
        col_indexes = pd.MultiIndex.from_product([elements, classes])
        categories_factors = pd.Index(categories_factors)
        col_indexes = col_indexes.map(lambda x: f"{x[0]} G{x[1]}")
        col_indexes = categories_factors.append(col_indexes)
        df_scores = pd.DataFrame(columns=col_indexes)

        features = self.features
        classes = self.class_names


        # features_wae = []
        # for feature in features:
        #   if feature not in ("Age", "Education"):
        #        features_wae.append(feature)

        i = 0
        for idx, feature in enumerate(features):
            explanation = ebm_global.data(features.index(feature))
            feature_cats = explanation["names"] if feature not in ("Age", "Education") else explanation["names"][:-1]
            scores = explanation["scores"]
            for idx_cat, cat in enumerate(feature_cats):
                for idx_cl, class_label in enumerate(classes):
                    up_err = explanation["upper_bounds"][idx_cat, idx_cl]
                    low_err = explanation["lower_bounds"][idx_cat, idx_cl]
                    df_scores.loc[i, categories_factors] = [feature, cat]
                    df_scores.loc[i, f"Score G{class_label}"] = scores[idx_cat, idx_cl]
                    df_scores.loc[i, f"Err Low G{class_label}"]  = low_err
                    df_scores.loc[i, f"Err High G{class_label}"]  = up_err
                i += 1
        df_scores["Factors"] = df_scores["Factors"].astype(float).astype(int)
        self.effects = df_scores
        return self.effects

    def plot_factor_effects(self, file_name):
        plotting.plot_effects(self.effects, file_name)

    def show_global_explanations(self):
        """Shows global explanations obtained with the EBM model."""
        show(self.global_exp)

    def get_top_imp_features(self, n_imp):
        """Returns the n_imp most important variables obtained from the EBM model.

        Args:
            n_imp: int
                The number of most important variables to return.
        Returns:
            top_imp_features: array(n_imp)
                The array containing the most important variables. The index
                corresponds to the variable labels and the values, the mean
                absolute logit.
        """
        feature_imp = pd.Series(self.feature_imp,
                                index=self.features,
                                name="Feature Importances")
        top_imp_features = feature_imp.sort_values(ascending=False)
        self.top_imp_features = top_imp_features[:n_imp]
        return self.top_imp_features

    def show_linear_eqs(self, save_dir, fig_name, type_features="Important"):
        """Plots the relationship between the dependent and independent
        variable.

        Args:
            top_features: array(n_imp)
                The list (variable labels) of most important variables to plot.
            save_dir: string
                The path of results saving.
            fig_name: string
                The name of the figure.
        Returns:
            None
        """
        plotting.plot_linear_eqs(self,
                                 save_dir,
                                 fig_name,
                                 type_features=type_features)

    def get_statistical_significance(self, top_features, file_names=None):
        """Returns the statistical significance test results (z-scores and p-values).

        Args:
            top_features: array(n_imp)
                The list (variable labels) of most important variables to plot.
            file_names: string
                    The name of files used to save the results.
        Returns:
            z_scores: array(n_bins, n_cogn_categories)
                Array of z-scores values where n_bins is the total number of
                bins of selected variables and n_cogn_categories the number of
                cognitive categories.
            p_values: array(n_bins, n_cogn_categories)
                Array of p-values corresponding to each bin.
        """
        z_scores = pd.DataFrame(columns=self.class_names)
        p_values = pd.DataFrame(columns=self.class_names)
        res_indexes = []
        i = 0
        for idx, feature in enumerate(top_features):
            idx_feat = self.features.index(feature)
            explanation = self.global_exp.data(idx_feat)
            scores = explanation["scores"]
            var_categories = explanation["names"]
            for idx_cat, cat in enumerate(var_categories):
                if cat == '0':
                    continue
                index_name = f"{feature} ({cat})" if cat != '1' else feature
                res_indexes.append(index_name)
                for idx_class, class_name in enumerate(self.class_names):
                    idx_cat = idx_cat if idx_cat < len(scores) else idx_cat - 1
                    cat_scores = scores[idx_cat]
                    score = cat_scores[idx_class]
                    # Indexing from second row because EBM associates 0 in first row of stds
                    std = self.standard_devs[idx_feat][idx_cat + 1][idx_class]
                    z_score = round(score / std, 3)
                    p_value = round(2 * stats.norm.cdf(-abs(z_score)), 3)
                    z_scores.loc[i, class_name] = z_score
                    p_values.loc[i, class_name] = p_value
                i += 1

        z_scores.index = res_indexes
        p_values.index = res_indexes
        if file_names:
            z_scores.to_csv(file_names["z_scores"] + ".csv")
            p_values.to_csv(file_names["p_values"] + ".csv")
            utils.save_pvalues(p_values, file_names["p_values"])
        self.z_scores = z_scores
        self.p_values = p_values
        return z_scores, p_values

    def plot_zscores(self, file_name):
        """Plots the obtained z_scores form the statistical significance analysis.

        Args:
            file_name: string,
                The name of file to save.
        Returns:
            None
        """
        if self.exp_type == TypeExp.WITH_AGE_EDUCATION:
            age_cats_to_plot = [60, 65, 70, 80,
                                85]  # Selected age bins to test significance
            edu_cats_to_plot = [
                6, 8, 10, 12, 14
            ]  # Selected education bins to test significance
            age_cat_label = [f"Age ({cat})" for cat in age_cats_to_plot]
            edu_cat_label = [f"Education ({cat})" for cat in edu_cats_to_plot]
            non_age_educ_cats = []
            for cat in self.z_scores.index:
                if "Age" not in cat and "Education" not in cat:
                    non_age_educ_cats.append(cat)
            final_cats = age_cat_label + edu_cat_label + non_age_educ_cats
            z_socres = self.z_scores.loc[final_cats]
        else:
            z_socres = self.z_scores
        plotting.plot_zscores(z_socres, file_name=file_name)

