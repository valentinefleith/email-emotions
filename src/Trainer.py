import pandas as pd
import numpy as np
from collections import namedtuple
import json
import joblib
from Classifier import Classifier

Metrics = namedtuple("Metrics", ["accuracy", "f1score"])


class Trainer:
    """
    A utility class to train and evaluate multiple classification models,
    compare their performance, and save the best hyperparameters and results.

    Attributes
    ----------
    models : list
        A list of model types (as strings) to be trained and compared.
    X : np.ndarray
        The feature matrix used for training and testing.
    y : np.ndarray
        The target variable.
    labels : list
        A list of class labels used in classification.
    """

    def __init__(self, models: list, X: np.ndarray, y: np.ndarray, labels: list):
        self.models = models
        self.X = X
        self.y = y
        self.labels = labels

    def compare_results(
        self, defined=False, save_results=True, save_best=True, table=True
    ):
        """
        Trains all models, evaluates them, and compares their results.

        Parameters
        ----------
        defined : bool, optional
            If True, uses predefined hyperparameters from JSON files (default is False).
        save_results : bool, optional
            If True, saves the results (classification reports and confusion matrices) to disk (default is True).
        save_best : bool, optional
            If True, saves the best-performing model to disk (default is True).
        table : bool, optional
            If True, prints a tabular comparison of model results (default is True).

        Returns
        -------
        None
        """
        overall_results = {}
        for model in self.models:
            print(f"Currently training {model}...")
            metrics = self.train_single_classifier(
                model, defined=defined, save_results=save_results
            )
            overall_results[model] = metrics
        if table:
            self.print_comparison(overall_results)
        if save_best:
            self.save_best_model(overall_results)

    def train_single_classifier(
        self, type: str, defined=False, save_results=True
    ) -> Metrics:
        """
        Trains and evaluates a single classification model.

        Parameters
        ----------
        type : str
            The type of model to train.
        defined : bool, optional
            If True, loads predefined hyperparameters from a JSON file (default is False).
        save_results : bool, optional
            If True, saves the results (classification reports and confusion matrices)(default is True).

        Returns
        -------
        Metrics
            A namedtuple containing the accuracy and F1-score of the model.
        """
        clf = Classifier(type, self.X, self.y, self.labels)
        if defined:
            with open(f"best_params/{type}.json", "r") as inf:
                best_params = json.load(inf)
        else:
            best_params = None
        y_hat = clf.classify(best_params=best_params)
        if save_results:
            clf.get_classificationReport(y_hat, save=True)
            clf.get_and_plot_confusionMatrix(y_hat, save=True, display=False)
        # else:
        # clf.get_classificationReport(y_hat, save=False)
        # clf.get_and_plot_confusionMatrix(y_hat, save=False, display=True)
        return Metrics(clf.accuracy, clf.f1_score)

    @staticmethod
    def print_comparison(overall_results: dict):
        """
        Prints a tabular comparison of the results for all models.

        Parameters
        ----------
        overall_results : dict
            A dictionary where keys are model types and values are their performance metrics.

        Returns
        -------
        None
        """
        results_df = pd.DataFrame.from_dict(overall_results)
        results_df = results_df.rename(index={0: "accuracy", 1: "f1-score"})
        print(f"\nOVERALL RESULTS:\n{results_df}")

    def get_best_params(self, save=True):
        """
        Performs a grid search to determine the best hyperparameters for each model.

        Parameters
        ----------
        save : bool, optional
            If True, saves the best parameters to JSON files in the "best_params" directory (default is True).

        Returns
        -------
        None
        """
        for model in self.models:
            print(f"Currently training {model}:")
            clf = Classifier(model, self.X, self.y, self.labels)
            best_params, results = clf.gridSearch(save=True)
            print(f"Best params: {best_params}")
            print(f"Train score: {results[0]:.2f}")
            print(f"Test score: {results[1]:.2f}")

    def save_best_model(self, overall_results: dict):
        """
        Saves the best model based on the highest accuracy from the provided results.

        This method selects the classifier with the highest accuracy from the
        `overall_results` dictionary, loads its best hyperparameters from a
        corresponding JSON file, and then saves the model with those parameters.

        Parameters
        ----------
        overall_results : dict
            A dictionary where the keys are model types (e.g., "SVM", "RFC", etc.) and the
            values are `Metrics` namedtuples containing the performance metrics (e.g., accuracy, f1-score)
            for each model. The model with the highest accuracy will be selected as the best model.

        NB
        --
        The best model's parameters are loaded from the `best_params` directory,
        using the model's type to find the corresponding JSON file. The trained model
        is then saved to the `saved_models` directory.
        """
        best_model_type = max(
            overall_results, key=lambda model: overall_results[model].accuracy
        )
        clf = Classifier(best_model_type, self.X, self.y, self.labels)
        with open(f"best_params/{best_model_type}.json", "r") as inf:
            best_params = json.load(inf)
        clf.save_model(best_params=best_params)
