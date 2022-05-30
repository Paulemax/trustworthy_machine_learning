import warnings
import scipy
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


warnings.filterwarnings('ignore')


def load_iris_into_dataframe() -> pd.DataFrame:
    iris = datasets.load_iris()
    return pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


def train_decision_tree(train: pd.DataFrame, test: pd.DataFrame, clf: DecisionTreeClassifier) \
    -> set[float, DecisionTreeClassifier]:

    def split(dataset: pd.DataFrame) -> set[pd.DataFrame, pd.DataFrame]:
        x = dataset.loc[::, ::-1]
        y = dataset.loc[::, "target"]
        return x, y

    train_x, train_y = split(train)
    test_x, test_y = split(test)
    clf.fit(train_x, train_y)
    score = clf.score(test_x, test_y)
    return score, clf


def part_one(train: pd.DataFrame, test: pd.DataFrame):
    clf = DecisionTreeClassifier(random_state=0)
    score, clf = train_decision_tree(train, test, clf)
    print(score)


def part_two(train: pd.DataFrame, test: pd.DataFrame, runs=10):
    classifiers = {}
    for random_state in range(runs):
        clf = DecisionTreeClassifier(random_state=random_state)
        score, clf = train_decision_tree(train, test, clf)
        classifiers[clf] = score

    scores = np.array(list(classifiers.values()))
    print(f"scores over {runs} runs: \nmean: {np.mean(scores)}\nstd: {np.std(scores)}")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def part_three(test: pd.DataFrame, bootstraps: int=10):
    clfs = []
    scores = []
    for b in range(bootstraps):
        samples = np.random.randint(0, len(test), size=len(test))
        x = test.iloc[samples, ::]
        y = test.iloc[samples, -1]
        clf = DecisionTreeClassifier() 
        clf.fit(x, y)
        clfs.append(clf)
        scores.append(clf.score(test.loc[::, ::-1], test.loc[::, "target"]))

    mean, lower, upper = mean_confidence_interval(scores)
    print(f"mean: {mean}, alpha1: {lower}, alpha2: {upper}")
    

def part_four(dataset: pd.DataFrame, cv: int=5):
    clf = DecisionTreeClassifier(random_state=42)
    x = dataset.loc[::, ::-1]
    y = dataset.loc[::, "target"]
    performance = cross_val_score(clf, x, y, cv=cv)
    print(f"performance of {cv} fold crossvalidation: {performance}")


if __name__ == "__main__":
    iris = datasets.load_iris()
    dataset = load_iris_into_dataframe()
    print(dataset)
    train, test = train_test_split(dataset, test_size=0.15, random_state=42) #, stratify=dataset["target"])    
    part_one(train, test)
    part_two(train, test, 50)
    part_three(test)
    part_four(dataset, cv=20)