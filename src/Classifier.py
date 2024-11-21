import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_predict, GridSearchCV


class Classifier:
    def __init__(self, type: str, X, y, labels):
        self.type = type
        self.X = X
        self.y = y
        self.labels = labels
        self.__gridSearch_methods = {
            "SVM": self.__gridSearch_SVM,
            "RFC": self.__gridSearch_RFC,
        }
        self.__defined_methods = {
            "SVM": self.__defined_SVM,
            "RFC": self.__defined_RFC,
        }
        self.__scoring = "accuracy"

    def gridSearch(self, save=False) -> tuple:
        best_params, scores = self.__gridSearch_methods[self.type]()
        if save:
            print(best_params)
            if not os.path.exists("best_params"):
                os.makedirs("best_params")
            with open(f"best_params/{self.type}.json", "w") as outf:
                json.dump(best_params, outf)
        return best_params, scores

    def __gridSearch_SVM(self):
        pass

    def __gridSearch_RFC(self):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2"],
        }
        clf = RandomForestClassifier()
        gs = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=self.__scoring)
        gs.fit(self.X, self.y)
        train_score = gs.best_score_
        test_score = gs.best_estimator_.score(self.X, self.y)
        return gs.best_params_, (train_score, test_score)

    def classify(self, best_params=None) -> np.ndarray:
        return self.__defined_methods[self.type](best_params)

    def __defined_SVM(self, best_params) -> np.ndarray:
        clf = (
            SVC(
                kernel=best_params["kernel"],
                probability=best_params["probability"],
                gamma=best_params["gamma"],
                C=best_params["C"],
            )
            if best_params is not None
            else SVC()
        )
        y_hat = cross_val_predict(clf, self.X, self.y, cv=10)
        return y_hat

    def __defined_RFC(self, best_params) -> np.ndarray:
        clf = (
            RandomForestClassifier(
                max_features=best_params["max_features"],
                n_estimators=best_params["n_estimators"],
                criterion=best_params["criterion"],
            )
            if best_params is not None
            else RandomForestClassifier()
        )
        y_hat = cross_val_predict(clf, self.X, self.y, cv=10)
        return y_hat

    def get_confusionMatrix(self, y_hat) -> np.ndarray:
        return confusion_matrix(self.y, y_hat)

    def get_and_plot_confusionMatrix(self, y_hat, save=False) -> np.ndarray:
        confMatrix = self.get_confusionMatrix(y_hat)
        fig, ax = plt.subplots(figsize=(12, 8))
        disp = ConfusionMatrixDisplay(confMatrix)
        disp.plot(cmap="OrRd", ax=ax)
        plt.title("Confusion matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)
        plt.tight_layout()
        plt.show()
        if save:
            if not os.path.exists("confusion_matrices"):
                os.makedirs("confusion_matrices")
            plt.savefig(f"confusion_matrices/{self.type}.png")
        return confMatrix

    def get_classificationReport(self, y_hat, save=False) -> str:
        report = classification_report(self.y, y_hat)
        print(report)
        if save:
            if not os.path.exists("classification_reports"):
                os.makedirs("classification_reports")
            with open(f"classification_reports/{self.type}.txt", "w") as outf:
                outf.write(str(report))
        return str(report)
