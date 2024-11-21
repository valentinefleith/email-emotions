import pandas as pd
import numpy as np
from collections import Counter
from Classifier import Classifier

PATH = "../data_featurized.csv"
LABEL_LIST = ["neutre", "colère", "tristesse", "joie", "surprise_pos", "surprise_neg"]
LABEL_DICT = {label: i for i, label in enumerate(LABEL_LIST)}


def get_data_arr(df):
    X = np.array(df[df.columns[:-2]])
    y = np.array(df["LABEL"].map(LABEL_DICT))
    return X, y


def get_stats(y):
    index_class_counts = Counter(y)
    for i, name in enumerate(LABEL_LIST):
        print(index_class_counts[i])
    str_class_counts = {
        name: index_class_counts[i] for i, name in enumerate(LABEL_LIST)
    }
    stats_df = pd.DataFrame(list(str_class_counts.items()), columns=["Class", "Count"])
    stats_df["Percentage"] = stats_df["Count"] / stats_df["Count"].sum() * 100
    print(stats_df)


def classify(type, X, y):
    classifier = Classifier(type, X, y, LABEL_LIST)
    y_hat = classifier.classify(best_params=None)
    classifier.get_classificationReport(y_hat, save=True)
    classifier.get_and_plot_confusionMatrix(y_hat, save=True)


def main():
    df = pd.read_csv(PATH)
    X, y = get_data_arr(df)
    get_stats(y)
    # classify("SVM", X, y)
    classify("RFC", X, y)


main()
