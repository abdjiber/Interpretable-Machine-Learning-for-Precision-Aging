from enum import IntEnum


class TypeExp(IntEnum):
    """Enumeration class used to run two types of experiments.

    When WITH_AGE_EDUCATION is specified, experiments are run with \
    with all covariates and Age, and Education. When WITHOUT_AGE_EDUCATION \
    the Age and Education variables are not considered in the experiments.
    """
    WITH_AGE_EDUCATION = 1
    WITHOUT_AGE_EDUCATION = 2