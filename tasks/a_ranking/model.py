from __future__ import annotations

import pandas as pd


def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """Train a model on ``train_data`` and predict rankings for ``test_data``.

    Args:
        train_data: Historical team-season data, including the target column.
        test_data: Team-season data for which rankings should be predicted.

    Returns:
        A dataframe containing predictions for ``test_data``.
    """
    
    if train_data.empty:
        raise ValueError("train_data must not be empty")
    if test_data.empty:
        raise ValueError("test_data must not be empty")

    # TODO: Select features and the target from train_data.
    # TODO: Fit the ranking model using the training data.
    # TODO: Apply the fitted model to test_data.
    raise NotImplementedError("The ranking model has not been implemented yet.")
