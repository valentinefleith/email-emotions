import os
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split


class Classifier:
    """
    Classification framework supporting SVM, Random Forest, Logistic Regression
    and Perceptron models.
    This class provides methods for:
    - hyperparameter tuning (GridSearch)
    - classification
    - evaluation (reports, confusion matrices)

    Attributes
    ----------
        public:
            type (str): The type of classifier to use ("SVM", "RFC", or "Perceptron").
            X (array-like): Feature matrix for training.
            y (array-like): Target vector for training.
            labels (list): List of label names for evaluation and display.
            accuracy (float): Accuracy score after cross-validation.
            f1-score (float): Macro f1-score after cross-validation.
        private:
            __gridSearch_methods (dict): Mapping of classifier types to their respective grid search methods.
            __scoring (str): The scoring metric to use during GridSearchCV (default: "accuracy").
    """

    def __init__(self, type: str, X, y, labels):
        self.type = type
        self.X = X
        self.y = y
        self.labels = labels
        self.accuracy = 0.0
        self.f1_score = 0.0
        self.__gridSearch_methods = {
            "SVM": self.__gridSearch_SVM,
            "RFC": self.__gridSearch_RFC,
            "Perceptron": self.__gridSearch_Perceptron,
            "LR": self.__gridSearch_LR,
        }
        self.__scoring = "accuracy"

    def gridSearch(self, save=False) -> tuple:
        """
        Performs a grid search to find the best hyperparameters for the specified classifier.

        Args
        ----
            save (bool): If True, saves the best parameters to a JSON file in the "best_params" directory.

        Returns
        -------
            tuple: A tuple containing the best hyperparameters (dict) and a tuple of train score (float).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        best_params, scores = self.__gridSearch_methods[self.type](
            X_train, X_test, y_train, y_test
        )
        if save:
            print(best_params)
            os.makedirs("best_params", exist_ok=True)
            with open(f"best_params/{self.type}.json", "w") as outf:
                json.dump(best_params, outf)
        return best_params, scores

    def __gridSearch_SVM(self, X_train, X_test, y_train, y_test):
        param_grid = {
            "kernel": ["rbf", "linear", "poly"],
            "C": [1, 10],
            "gamma": [0.01, 0.1, 1, "scale"],
        }
        clf = SVC()
        gs = GridSearchCV(
            clf, param_grid=param_grid, cv=5, scoring=self.__scoring, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_model = SVC(**gs.best_params_).fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        return gs.best_params_, (gs.best_score_, test_score)

    def __gridSearch_Perceptron(self, X_train, X_test, y_train, y_test):
        param_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.001, 0.0001],
        }
        clf = Perceptron()
        gs = GridSearchCV(
            clf, param_grid=param_grid, cv=5, scoring=self.__scoring, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_model = Perceptron(**gs.best_params_).fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        return gs.best_params_, (gs.best_score_, test_score)

    def __gridSearch_LR(self, X_train, X_test, y_train, y_test):
        param_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.1, 1.0, 10],
            "max_iter": [100, 200],
        }
        clf = LogisticRegression()
        gs = GridSearchCV(
            clf, param_grid=param_grid, cv=5, scoring=self.__scoring, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_model = LogisticRegression(**gs.best_params_).fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        return gs.best_params_, (gs.best_score_, test_score)

    def __gridSearch_RFC(self, X_train, X_test, y_train, y_test):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2"],
        }
        clf = RandomForestClassifier(class_weight="balanced")
        gs = GridSearchCV(
            clf, param_grid=param_grid, cv=5, scoring=self.__scoring, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_model = RandomForestClassifier(**gs.best_params_).fit(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        return gs.best_params_, (gs.best_score_, test_score)

    def classify(self, best_params=None, cv=5):
        """
        Performs classification using the specified classifier and optional best hyperparameters.

        Args
        ----
            best_params (dict, optional): Best hyperparameters for the classifier.
            cv (int): Number of cross-validation folds (default: 5).

        Returns
        -------
            array: Predicted labels from cross-validation.
        """
        if self.type == "SVM":
            clf = SVC(**best_params) if best_params else SVC()
        elif self.type == "RFC":
            clf = (
                RandomForestClassifier(**best_params)
                if best_params
                else RandomForestClassifier()
            )
        elif self.type == "Perceptron":
            clf = Perceptron(**best_params) if best_params else Perceptron()
        elif self.type == "LR":
            clf = (
                LogisticRegression(**best_params)
                if best_params
                else LogisticRegression()
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.type}")
        # Example of using threshold for out of scope category commented below:
        # if len(self.labels) != 9:
        #     y_hat = self.__perform_threshold_classif(clf)
        y_hat = cross_val_predict(clf, self.X, self.y, cv=cv)
        self.accuracy = accuracy_score(self.y, y_hat)
        self.f1_score = f1_score(self.y, y_hat, average="macro")
        return y_hat

    def __perform_threshold_classif(self, clf) -> np.ndarray:
        """
        More exprimental part for `out of scope` class.
        TODO:
        - improve threshold selection
        - using it implies changing the training dataset by reducing the number of categories since out of scope would be a separated category.
        """
        predicted_probs = cross_val_predict(
            clf, self.X, self.y, cv=cv, method="predict_proba"
        )
        y_hat = []
        threshold = np.mean([max(prob) for prob in predicted_probs])
        print(f"THRESHOLD = {threshold}")
        print(predicted_probs)
        for prob in predicted_probs:
            predicted_class = np.argmax(prob)
            max_prob = max(prob)
            if max_prob < threshold:
                y_hat.append(5)
            else:
                y_hat.append(predicted_class)
        y_hat = np.array(y_hat)
        return y_hat

    def save_model(self, best_params=None):
        """
        Trains and saves the best model based on the specified classifier type and hyperparameters.

        Parameters
        ----------
        best_params : dict, optional
            A dictionary of hyperparameters to be used for initializing the model.
            If None, the classifier is initialized with default hyperparameters.
            Default is None.

        Raises
        ------
        ValueError
            If the classifier type specified in `self.type` is not supported.

        NB
        --
        The saved model and metadata will be stored in the `saved_models` directory.
        The model file will be named based on the classifier type (e.g., `SVM_best.joblib`),
        and the metadata will be stored in a corresponding JSON file (e.g., `SVM_metadata.json`).
        """
        if self.type == "SVM":
            clf = SVC(**best_params) if best_params else SVC()
        elif self.type == "RFC":
            clf = (
                RandomForestClassifier(**best_params)
                if best_params
                else RandomForestClassifier()
            )
        elif self.type == "Perceptron":
            clf = Perceptron(**best_params) if best_params else Perceptron()
        elif self.type == "LR":
            clf = (
                LogisticRegression(**best_params)
                if best_params
                else LogisticRegression()
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.type}")
        clf.fit(self.X, self.y)
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(clf, f"saved_models/{self.type}_best.joblib")
        metadata = {
            "type": self.type,
            "best_params": best_params,
            "labels": self.labels,
        }
        with open(f"saved_models/{self.type}_metadata.json", "w") as outf:
            json.dump(metadata, outf)

    def get_confusionMatrix(self, y_hat) -> np.ndarray:
        """
        Computes the confusion matrix for the given predictions.

        Args
        ----
            y_hat (array-like): Predicted labels.

        Returns
        -------
            np.ndarray: Confusion matrix.
        """
        return confusion_matrix(self.y, y_hat)

    def get_and_plot_confusionMatrix(
        self, y_hat, save=False, display=True
    ) -> np.ndarray:
        """
        Computes and plots the confusion matrix.

        Args
        ----
            y_hat (array-like): Predicted labels.
            save (bool): If True, saves the confusion matrix plot to the "results" directory.
            display (bool): If True, displays the confusion matrix plot.

        Returns
        -------
            np.ndarray: Confusion matrix.
        """
        confMatrix = self.get_confusionMatrix(y_hat)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confMatrix)
        disp.plot(cmap="OrRd", ax=ax)
        plt.title("Confusion matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        ax.set_xticklabels(self.labels, rotation=45)
        ax.set_yticklabels(self.labels)
        plt.tight_layout()
        if display:
            plt.show()
        if save:
            os.makedirs(f"results/{self.type}", exist_ok=True)
            plt.savefig(f"results/{self.type}/confusion_matrix.png")
        return confMatrix

    def get_classificationReport(self, y_hat, save=False) -> str:
        """
        Generates and optionally saves the classification report.

        Args
        ----
            y_hat (array-like): Predicted labels.
            save (bool): If True, saves the classification report to the "results" directory.

        Returns
        -------
            str: The classification report.
        """
        report = classification_report(self.y, y_hat)
        print(report)
        if save:
            os.makedirs(f"results/{self.type}", exist_ok=True)
            with open(f"results/{self.type}/classification_report.txt", "w") as outf:
                outf.write(str(report))
        return str(report)
