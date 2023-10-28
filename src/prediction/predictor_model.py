import os
import warnings
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the HistGradientBoosting regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "HistGradientBoosting_regressor"

    def __init__(
        self,
        loss: Optional[str] = 'squared_error',
        learning_rate: Optional[float] = 0.1,
        max_depth: Optional[Union[int, None]] = None,
        max_leaf_nodes: Optional[Union[int, None]] = 31,
        min_samples_leaf: Optional[int] = 20,
        **kwargs,
    ):
        """Construct a new HistGradientBoosting regressor.

        Args:
            loss (optional, str):
            {‘squared_error’, ‘absolute_error’, ‘gamma’, ‘poisson’, ‘quantile’}, default=’squared_error’
            The loss function to use in the boosting process. Note that the “squared error”,
            “gamma” and “poisson” losses actually implement “half least squares loss”, “half gamma deviance” and “half
            poisson deviance” to simplify the computation of the gradient. Furthermore, “gamma” and “poisson” losses
            internally use a log-link, “gamma” requires y > 0 and “poisson” requires y >= 0. “quantile” uses the pinball
            loss.

            learning_rate (optional, float): The learning rate, also known as shrinkage. This is used as a
            multiplicative factor for the leaves values. Use 1 for no shrinkage.

            max_depth (optional, int, None): The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf. Depth isn’t constrained by default.

            max_leaf_nodes (optional, int, None): The maximum number of leaves for each tree. Must be strictly
            greater than 1. If None, there is no maximum limit.

            min_samples_leaf (optional, int): The minimum number of samples per leaf. For small datasets with less
            than a few hundred samples, it is recommended to lower this value since only very shallow trees would be
            built.

        """
        self.loss = loss,
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes,
        self.min_samples_leaf = min_samples_leaf
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> HistGradientBoostingRegressor:
        """Build a new HistGradientBoosting regressor."""
        model = HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=0
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the HistGradientBoosting regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The targets of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression target for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression target.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the HistGradientBoosting regressor and return coefficient of
        determination (r-squared) of the prediction.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The target of the test data.
        Returns:
            float: The coefficient of determination of the prediction of the Random
                   Forest regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the HistGradientBoosting regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the HistGradientBoosting regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            regressor: A new instance of the loaded HistGradientBoosting regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"learning_rate: {self.learning_rate}, "
            f"loss: {self.loss}, "
            f"max_depth: {self.max_depth}, "
            f"max_leaf_nodes: {self.max_leaf_nodes}, "
            f"min_samples_leaf: {self.min_samples_leaf})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_inputs (pd.DataFrame): The training data inputs.
        train_targets (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the accuracy.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The accuracy of the regressor model.
    """
    return model.evaluate(x_test, y_test)
