import pandas as pd
import numpy as np
from Classifier import Classifier

PATH = "../data_featurized.csv"
LABEL_LIST = ["neutre", "colère", "tristesse", "joie", "surprise_pos", "surprise_neg"]
LABEL_DICT = {label: i for i, label in enumerate(LABEL_LIST)}


def get_data_arr(df):
    X = np.array(df[df.columns[:-2]])
    y = np.array(df["LABEL"].map(LABEL_DICT))
    return X, y


def main():
    df = pd.read_csv(PATH)
    X, y = get_data_arr(df)
    print(X.shape)
    print(y.shape)
    svm_classifier = Classifier("SVM", X, y, LABEL_LIST)
    y_hat = svm_classifier.classify(best_params=None)
    svm_classifier.get_classificationReport(y_hat, save=True)
    svm_classifier.get_and_plot_confusionMatrix(y_hat, save=True)


main()
