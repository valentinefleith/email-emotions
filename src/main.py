import pandas as pd
import numpy as np
from collections import Counter
from Classifier import Classifier
from Trainer import Trainer
from sklearn.utils import resample

PATH = "data/data_featurized_full_sw.csv"
LABEL_LIST = ["neutre", "colère", "tristesse", "joie", "surprise_pos", "surprise_neg"]
LABEL_DICT = {label: i for i, label in enumerate(LABEL_LIST)}


def get_data_arr(df):
    X = np.array(df[df.columns[:-3]])
    y = np.array(df["LABEL"].map(LABEL_DICT))
    return X, y


def get_stats(y):
    index_class_counts = Counter(y)
    str_class_counts = {
        name: index_class_counts[i] for i, name in enumerate(LABEL_LIST)
    }
    stats_df = pd.DataFrame(list(str_class_counts.items()), columns=["Class", "Count"])
    stats_df["Percentage"] = stats_df["Count"] / stats_df["Count"].sum() * 100
    print(stats_df)


def downsample_data(df):
    new_df = df.query("`AI` == 1")
    human_generated = df.query("`AI` == 0")
    for label in LABEL_LIST:
        this_label_data = human_generated.query("`LABEL` == @label")
        resampled = resample(
            this_label_data, replace=False, n_samples=15, random_state=42
        )
        new_df = pd.concat([new_df, resampled])
    return new_df.reset_index(drop=True)


def classify(type, X, y):
    classifier = Classifier(type, X, y, LABEL_LIST)
    y_hat = classifier.classify(best_params=None)
    classifier.get_classificationReport(y_hat, save=True)
    classifier.get_and_plot_confusionMatrix(y_hat, save=True)


def main():
    df = pd.read_csv(PATH)
    new_df = downsample_data(df)
    print(new_df)
    new_df.to_csv("data/balanced.csv", sep="|")
    X, y = get_data_arr(new_df)
    trainer = Trainer(["SVM", "RFC", "LR", "Perceptron"], X, y, LABEL_LIST)
    trainer.get_best_params(save=True)
    trainer.compare_results(save_results=True, defined=True, save_best=False)
    # get_stats(y)
    # classify("SVM", X, y)
    # classify("RFC", X, y)


main()
