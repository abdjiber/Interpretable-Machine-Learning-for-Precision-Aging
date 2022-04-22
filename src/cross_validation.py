import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class CrossValidate:
    def __init__(self, n_splits, random_state, verbose=False):
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose

    def get_models(self):
        EBM = ExplainableBoostingClassifier
        LR = LogisticRegression
        MLP = MLPClassifier
        XGB = XGBClassifier
        models = {
            "EBM": EBM(n_jobs=-1),
            "LR": LR(n_jobs=-1),
            "MLP": MLP(),
            "XGB": XGB(n_jobs=-1),
        }
        return models

    def fit(self, X, y):
        scores = self.cv(X, y)
        return scores

    def roc_score(self, y_true, y_pred):
        y_true = pd.get_dummies(y_true)
        score = roc_auc_score(y_true, y_pred)
        return score

    def cv(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits,
                              random_state=self.random_state)
        models = self.get_models()
        n_models = len(models)
        init_scores = np.zeros((n_models, 2), dtype=float)
        roc_scores = pd.DataFrame(init_scores,
                                  columns=["Mean", "STD"],
                                  index=models.keys())
        for i, (model_name, model) in enumerate(models.items()):
            scores = np.zeros((self.n_splits, ), dtype=float)
            for k, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                x_train, y_train = X.iloc[train_idx], y[train_idx]
                x_val, y_val = X.iloc[val_idx], y[val_idx]
                model.fit(x_train, y_train)
                y_pred = model.predict_proba(x_val)
                auc = self.roc_score(y_val, y_pred)
                scores[k] = auc
                if self.verbose:
                    print(f"Model {model_name} Fold {k+1} Score {auc}")
            roc_scores.loc[model_name] = (scores.mean().round(4),
                                          scores.std().round(3))
        return roc_scores
