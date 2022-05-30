import math
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_dataset(name:str ="x") -> pd.DataFrame:
    np.random.seed(42)
    if(name=="x"):
        pos = 1000
        neg = 10000
    elif(name=="y"):
        pos = 10000
        neg = 1000
    else:
        pos = 10000
        neg = 10000
    label = np.concatenate((np.ones(pos),np.zeros(neg)))
    np.random.shuffle(label)
    
    preda = []
    predb = []
    
    for i,l in enumerate(label):
        if(l==0):
            preda.append(np.random.beta(2,3))
            predb.append(np.random.beta(2,3))
        else:
            predb.append(np.random.beta(4,3))
            if(np.sum(label[:i+1])<0.3*pos):
                preda.append(np.random.beta(12,2))
            else:
                preda.append(np.random.beta(3,4))
    return pd.DataFrame({"A":preda, "B":predb, "label":label})


def generate_datasets() -> set[str, pd.DataFrame]:
    datasets = {}
    for label in ("x", "y", "z"):
        datasets[label] = generate_dataset(name=label)
    return datasets


def eval_classification(dataset: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    # evaluate the dataset to either 0 or 1
    dataset = dataset.copy() # lets keep our dataset immutable in here to avoid confusion
    dataset[dataset >= threshold] = 1
    dataset[dataset < threshold] = 0
    return dataset


def part_one(dataset: pd.DataFrame):
    eval = eval_classification(dataset)

    acc: Callable[[pd.Series, pd.Series], float] = lambda a, b : np.sum(a==b) / len(a)
    print (f"""accuracy:  
        classifier A: {acc(eval['A'], eval['label'])}
        classifier B: {acc(eval["B"], eval["label"])}""")

    preccision: Callable[[pd.DataFrame, pd.DataFrame], float] = lambda tp, fp : len(tp) / (len(tp) + len(fp))
    recall: Callable[[pd.DataFrame, pd.DataFrame], float] = lambda tp, fn : len(tp) / (len(tp) + len(fn))

    pos = eval[eval["label"] == 1]
    neg = eval[eval["label"] == 0]

    precc_a = preccision(pos[pos["label"] == pos["A"]], neg[neg["A"] != neg["label"]])
    precc_b = preccision(pos[pos["label"] == pos["B"]], neg[neg["B"] != neg["label"]])

    rec_a = recall(pos[1 == pos["A"]], pos[pos["A"] != 1])
    rec_b = recall(pos[1 == pos["B"]], pos[pos["B"] != 1])

    print(precc_a, precc_b)
    print(rec_a, rec_b)

    def f_score(prec:float, rec:float, beta:int=1):
        return (beta**2 + 1) * ((prec * rec) / (beta**2 * prec + rec))

    print(f_score(precc_a, rec_a))
    print(f_score(precc_b, rec_b))


def get_true_positives(dataset: pd.DataFrame, label: str) -> int:
    pos = dataset[dataset["label"] == 1]
    tp = pos[pos[label] == 1]
    return len(tp)


def get_false_positives(dataset: pd.DataFrame, label: str) -> int:
    """ extracts the false positives from a given dataset
        false posives = positive results that should be negative
    """
    neg = dataset[dataset["label"] == 0]
    fp = neg[neg[label] == 1]
    return len(fp)


def get_false_negatives(dataset: pd.DataFrame, label: str) -> int:
    """ extract the amount of false negative results from a given dataset
        false negatives = negative results that should be positive
    """
    neg = dataset[dataset["label"] == 1]
    fn = neg[neg[label] == 0]
    return len(fn)


def get_true_negatives(dataset: pd.DataFrame, label: str) -> int:
    neg = dataset[dataset["label"] == 0]
    tn = neg[neg[label] == 0]
    return len(tn)


def get_confusion_matrix(dataset: pd.DataFrame, label: str, threshold: float) -> dict[str, int]:
    cm = {}
    cm["tp"] = get_true_positives(dataset, label) 
    cm["fp"] = get_false_positives(dataset, label) 
    cm["fn"] = get_false_negatives(dataset, label)
    cm["tn"] = get_true_negatives(dataset, label)
    cm["t"] = threshold
    return cm


def get_sensitivity(confusion_matrix: dict[str, int]) -> float:
    return confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fn"])


def get_specificity(confusion_matrix: dict[str, int]) -> float:
    return confusion_matrix["tn"] / (confusion_matrix["tn"] + confusion_matrix["fp"])


def get_precision(confusion_matrix: dict[str, int]) -> float:
    return confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"])


def calc_mcc_fone_curve(dataset: pd.DataFrame, thresholds: np.ndarray, label: str) -> set[np.ndarray, np.ndarray]:
    un_mcc_scores = []
    fone_scores = []
    for threshold in thresholds:
        eval = eval_classification(dataset, threshold)
        cm = get_confusion_matrix(eval, label)

        if(cm["tp"] == 0):
            continue

        un_mcc_scores.append(unit_normalized_mcc_score(cm))
        fone_scores.append(f_score(get_precision(cm), get_sensitivity(cm), beta=1))

    return np.array(un_mcc_scores), np.array(fone_scores)


def mcc_fone_curve(confucion_matrixes: dict[float, dict[str, int]]) -> set[np.ndarray, np.ndarray]:
    un_mcc_scores = []
    fone_scores = []
    for _, cm in confucion_matrixes.items():
        if(cm["tp"] == 0):
            continue

        un_mcc_scores.append(unit_normalized_mcc_score(cm))
        fone_scores.append(f_score(get_precision(cm), get_sensitivity(cm), beta=1))

    return np.array(un_mcc_scores), np.array(fone_scores)


def precision_recall_curve(confucion_matrixes: dict[float, dict[str, int]]) -> set[np.ndarray, np.ndarray]:
    precision:list[float] = []
    recall:list[float] = []
    for _, cm in confucion_matrixes.items():

        if(cm["tp"]) == 0:
            continue

        precision.append(get_precision(cm))
        recall.append(get_sensitivity(cm))

    return np.array(precision), np.array(recall)


def get_precision_recall_curve(dataset: pd.DataFrame, thresholds: np.ndarray, label: str) -> set[np.ndarray, np.ndarray]:
    precision = []
    recall = []
    for threshold in thresholds:
        eval = eval_classification(dataset, threshold)
        cm = get_confusion_matrix(eval, label)

        if(cm["tp"] == 0):
            continue

        precision.append(get_precision(cm))
        recall.append(get_sensitivity(cm))

    return np.array(precision), np.array(recall)


def roc_curve(confucion_matrixes: dict[float, dict[str, int]]) -> set[np.ndarray, np.ndarray]:
    fp_rates:list[float] = []
    tp_rates:list[float] = []
    for t, cm in confucion_matrixes.items():
        tp_rates.append(get_sensitivity(cm))
        fp_rates.append(1 - get_specificity(cm))       
    return np.array(fp_rates), np.array(tp_rates)


def get_roc_curve(dataset: pd.DataFrame, thresholds: np.ndarray, label: str) -> set[np.ndarray, np.ndarray]:
    fp_rates:list[float] = []
    tp_rates:list[float] = []
    for threshold in thresholds: 
        eval = eval_classification(dataset, threshold)
        cm = get_confusion_matrix(eval, label)

        tp_rates.append(get_sensitivity(cm))
        fp_rates.append(1 - get_specificity(cm))

    return np.array(fp_rates), np.array(tp_rates)


def f_score(prec: float, rec: float, beta: float=1) -> float:
    return (beta**2 + 1) * ((prec * rec) / (beta**2 * prec + rec))


def mcc_score(confusion_matrix: set[str, int]) -> float:
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    tn = confusion_matrix["tn"]
    fn = confusion_matrix["fn"]
    return ((tp * tn) - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def unit_normalized_mcc_score(confusion_matrix: set[str, int]) -> float:
    return (mcc_score(confusion_matrix) + 1) / 2


def calc_auc(x: np.ndarray, y: np.ndarray) -> float:
    # sort tp rates
    sorted_indices = np.argsort(y)
    sx = x[sorted_indices]
    sy = y[sorted_indices]

    auc = 0
    for i in range(1, len(sy)):
        d = sx[i] - sx[i - 1]
        auc += d * sy[i]
    return auc


def calc_aupr(x: np.ndarray, y: np.ndarray) -> float:
    # sort tp rates
    sorted_indices = np.argsort(x)
    sx = x[sorted_indices]
    sy = y[sorted_indices]

    auc = 0
    for i in range(1, len(sy)):
        d = sx[i] - sx[i - 1]
        auc += d * sy[i]
    return auc




def get_mcc_fone_score(mcc_scores: np.ndarray, fone_scores: np.ndarray, thresholds: np.ndarray)-> set[float, float, float]:
    i = np.argmin(np.sqrt((mcc_scores - 1) ** 2 + (fone_scores - 1) ** 2))
    return mcc_scores[i], fone_scores[i], thresholds[i]


def part_two(dataset: pd.DataFrame):
    thresholds = np.arange(0.1, 1, 0.01)
    fig = plt.figure(figsize=(1, 2))
    aucs = []
    for i, label in enumerate(("A", "B")) :
        fp_rates, tp_rates = get_roc_curve(dataset, thresholds, label)
        aucs.append(calc_auc(fp_rates, tp_rates))
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(f"dataset: {label}")
        ax.set_ylabel("true positive rate")
        ax.set_xlabel("false positive rate")
        ax.plot(fp_rates, tp_rates)

    print(aucs)
    plt.show()

def part_three(dataset: pd.DataFrame):
    thresholds = np.arange(0.1, 1, 0.01)
    fig = plt.figure(figsize=(1, 2))
    for i, label in enumerate(("A", "B")):
        un_mcc_scores, fone_scores = calc_mcc_fone_curve(dataset, thresholds, label)
        print(f"mcc f1 score: {get_mcc_fone_score(un_mcc_scores, fone_scores, thresholds)}")
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(f"dataset: {label}")
        ax.set_ylabel("unit normalized mcc score")
        ax.set_xlabel("f1 score")
        ax.plot(fone_scores, un_mcc_scores)

    plt.show()

    # precision recall curve
    fig = plt.figure(figsize=(1, 2))
    for i, label in enumerate(("A", "B")):
        precision, recall = get_precision_recall_curve(dataset, thresholds, label)
        print(f"aupr score: {calc_aupr(x=recall, y=precision)}")
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_title(f"dataset: {label}")
        ax.set_ylabel("precision")
        ax.set_xlabel("recall")
        ax.plot(recall, precision)

    plt.show()


def get_cms(dataset: pd.DataFrame, thresholds: np.ndarray, label: str) -> dict[float, dict[str, int]]:
    # meh float as dict key is kinda bad
    cms = {}
    for threshold in thresholds:
        eval = eval_classification(dataset, threshold)
        cms[threshold] = get_confusion_matrix(eval, label, threshold)

    return cms

def part_four(dataset: pd.DataFrame):
    train, test = train_test_split(dataset, test_size=0.1, stratify=dataset["label"], random_state=42)
    thresholds = np.arange(0.1, 1, 0.01)
    cms = get_cms(test, thresholds, "A")
    f_one_scores = []
    ts = []
    for t, cm in cms.items():
        if cm["tp"] == 0:
            continue
        f_one_scores.append(f_score(get_precision(cm), get_sensitivity(cm), beta=1))
        ts.append(t)
    
    i = np.argmax(np.array(f_one_scores))
    print(f"best f1 score on test set with threshold: {ts[i]}, with score: {f_one_scores[i]}")
    # get cm for train set with threshold ts[i]
    train_cm = get_confusion_matrix(eval_classification(train, threshold=ts[i]), "A")
    default_cm = get_confusion_matrix(eval_classification(train), "A")

    print(f"f1 score for data_set with threshold: {ts[i]:.2f}, with score: \
        {f_score(get_precision(train_cm), get_sensitivity(train_cm), beta=1)}")
    print(f"f1 score for data_set with threshold: 0.5, with score: \
        {f_score(get_precision(default_cm), get_sensitivity(default_cm), beta=1)}")


if __name__ == '__main__' :
    # dataset = generate_dataset()
    # part_one(dataset)
    # part_two(dataset)
    # part_three(dataset)
    # part_four(dataset)
    pass