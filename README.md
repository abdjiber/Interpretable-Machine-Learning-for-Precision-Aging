This repository contains the code source of experiments performed in the paper "Impact of daily lifestyle activities on the cognitive health of older adults: An interpretable machine learning approach" submitted at NeurIPS 2022". In this paper, we used the Explainable Boosting Machine (EBM) model to assess the relationship between background factors (Age and Education), daily lifestyle activities, and the cognitive health of older adults.

The modules are described as follows:

- experiments.py: contains the main code to run all experiments.
- src/explain.py: contains the Explain class used to provide the following explanations
  - Feature importance (Mean absolute logit from the EBM model).
  - Weights associated to each variable bin. These weights are used to plot a linear relationship between the dependent and independent variables.
  - Statistical significance scores: z-scores and p-values.
- src/plotting.py: contains utility functions for plotting.
- src/utils.py: contains addtional utility functions.
- src/type_experiments.py: contains a class used to specify the type of experiments to run.
- src/cross_validation.py: contains a class used to cross-validate selected machine learning models. This module is used to compare the performance of the EBM, Logistic Regression, XGBoost, and Multi-Layer Perceptrons.
- src/config.py: contains experimental settings (data set name, plotting parameters, etc...).

All experiements in our paper are reproducible. To that end, please follow the instructions below.

1. Dowload the 2012 and 2016 waves of Health and Retirement Study dataset [here](https://hrs.isr.umich.edu/).
2. Concatenate the two waves so that each individual in the dataset has two observations (2012 and 2016).
3. Drop all missing values and filter the dataset by selecting only individuals with age > 65.
4. Select the variables used in the experiments:
   - Immadiate and delayed cognitive scores.
   - Background factors: Age and Education.
   - Daily lifestyle activities (see Table A.2) in the paper.
5. Process the selected variables:
   - Cognitive scores: create a composite score (Immadite + Delayed scores) and stratify the obtain scores into 3, 5, and 9 cognitive categories.
   - Binarize the daily lifestyle activities with the process described in Section 3.1 in the paper.
6. Install all package dependencies:

```sh
pip install -r requirements.txt
```

7. Set the experiment configurations (dataset name, saving paths in your computer) in the config.py file.
8. Run the experiments with:

```python
python experiments.py
```

If you would like to do the performance comparison analysis, you can specify the --perf_comp flag as follows:

```python
python experiments.py --perf_comp
```
