import os
import warnings

import numpy as np
from absl import app, flags

from src.config import (BASE_PATH_DS, DS_COGN_3_9_CATS, DS_COGN_5_CATS,
                        N_IMP_FEATURES, PATH_SAVE, RANDOM_SEED)
from src.cross_validation import CrossValidate
from src.explain import Explain
from src.type_experiments import TypeExp
from src.utils import stat_comparisons, set_plotting_params, load_data

FLAGS = flags.FLAGS
warnings.filterwarnings(action="ignore")
np.random.seed(RANDOM_SEED)

flags.DEFINE_boolean(
    'explain', False,
    'Runs the EBM model with different experiment settings to explain\
         the relationship between the dependent variable and covariates.')
flags.DEFINE_boolean(
    'perf_comp', False,
    'Performs a 5-Fold crossvalidation to compare the EBM model with a logistic\
        regression model, XGBoost, and Multi-Layer Perceptrons.')


def run_experiment(type_exp, n_cogn_cats):
    cogn_exp = f"cogn_{n_cogn_cats}_cats"
    save_dir = os.path.join(PATH_SAVE, cogn_exp)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if type_exp == TypeExp.WITH_AGE_EDUCATION:
        suffix_file_names = f"{cogn_exp}_with_AE"
    else:
        suffix_file_names = f"{cogn_exp}_without_AE"

    if n_cogn_cats == 5:
        full_path_ds = os.path.join(BASE_PATH_DS, DS_COGN_5_CATS)
        output_var_name = "Cogn_Strat_5"
        cols_to_drop = None
    else:
        full_path_ds = os.path.join(BASE_PATH_DS, DS_COGN_3_9_CATS)
        print(full_path_ds)
        if n_cogn_cats == 3:
            output_var_name = "Cogn_Strat_3"
            cols_to_drop = ["Cogn_Strat_9", "Cogn_Strat_5"]
        else:
            output_var_name = "Cogn_Strat_9"
            cols_to_drop = ["Cogn_Strat_3", "Cogn_Strat_5"]

    X, y = load_data(full_path_ds, output_var_name, drop_cols=cols_to_drop)
    if type_exp == TypeExp.WITHOUT_AGE_EDUCATION:
        X = X.drop(["Age", "Education"], axis=1)

    suffix_file_names += "_all_vars"
    explainer = Explain(X, y, type_exp=type_exp)
    explainer.fit()
    
    effects_file_names = os.path.join(save_dir, "effects" + f"_{suffix_file_names}")
    effects = explainer.get_factor_effects()

    set_plotting_params()
    explainer.plot_factor_effects(effects_file_names)
    effects.to_csv(f"{effects_file_names}.csv", header=True, index=False)

    top_imp_features = explainer.get_top_imp_features(N_IMP_FEATURES)
    # explainer.show_linear_eqs(save_dir=save_dir,
    #                          fig_name=f"linear_eqs_{suffix_file_names}.png",
    #                          type_features="All")
    stats_file_names = {
        "z_scores": os.path.join(save_dir, f"z_scores_{suffix_file_names}_new"),
        "p_values": os.path.join(save_dir, f"p_values_{suffix_file_names}_new"),
    }
    _ = explainer.get_statistical_significance(
        explainer.features,  # explainer features,
        file_names=stats_file_names)
    # file_name_zscores = os.path.join(save_dir, f"z_scores_{suffix_file_names}")
    # explainer.plot_zscores(file_name=file_name_zscores)

    # Running performance comparison if the flag cperf_comp is true.
    if FLAGS.perf_comp:
        cv = CrossValidate(n_splits=5, random_state=RANDOM_SEED, verbose=False)
        (roc_auc_scores, duration_models, scores_kf) = cv.fit(X, y)

        path_save_roc_auc = os.path.join(save_dir,
                                         f"roc_auc_{suffix_file_names}_new")
        roc_auc_scores.to_csv(path_save_roc_auc + "_new.csv", index=True)
        stat_comparisons(scores_kf, path_save_roc_auc)

        path_save_duration = os.path.join(save_dir,
                                          f"duration_{suffix_file_names}")
        duration_models.to_csv(path_save_duration + "_new.csv", index=True)


def main(argv):
    run_experiment(type_exp=TypeExp.WITH_AGE_EDUCATION, n_cogn_cats=3)
    run_experiment(type_exp=TypeExp.WITHOUT_AGE_EDUCATION, n_cogn_cats=3)

    run_experiment(type_exp=TypeExp.WITH_AGE_EDUCATION, n_cogn_cats=5)
    run_experiment(type_exp=TypeExp.WITHOUT_AGE_EDUCATION, n_cogn_cats=5)

    run_experiment(type_exp=TypeExp.WITH_AGE_EDUCATION, n_cogn_cats=9)
    run_experiment(type_exp=TypeExp.WITHOUT_AGE_EDUCATION, n_cogn_cats=9)

if __name__ == '__main__':
    app.run(main)
