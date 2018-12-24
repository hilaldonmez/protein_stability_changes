from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek


# %%
def get_result_label(file_name):
    result_label = []
    with open(file_name) as f:
        for line in f:
            result_label.append(line.split()[0])
    return result_label


def train_random_forest(X, y, n_splits):
    clf = RandomForestClassifier(n_estimators=100)
    calculate_metrics(clf, X, y, n_splits)


def train_svm(X, y, n_splits):
    clf = svm.SVC(C=5, gamma=0.1, kernel='rbf')
    calculate_metrics(clf, X, y, n_splits)


def cross_validation(clf, X, y, label, n_splits):
    kf = KFold(n_splits)
    kf.get_n_splits(X)
    accuracy = 0
    positive_precision = 0
    positive_recall = 0
    negative_precision = 0
    negative_recall = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        smt = SMOTETomek(ratio='auto')
        X_smt, y_smt = smt.fit_sample(X_train, y_train)
        clf.fit(X_smt, y_smt)

        predicted_y = clf.predict(X_test)
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i, j in zip(predicted_y, y_test):
            if i == label and j == label:
                tp += 1
            elif j == label:
                fn += 1
            elif i == label:
                fp += 1
            else:
                tn += 1

        accuracy += (tp + tn) / (tp + fn + fp + tn)
        positive_precision += 0 if tp + fp == 0 else tp / (tp + fp)
        positive_recall += 0 if tp + fn == 0 else tp / (tp + fn)
        negative_precision += 0 if tn + fn == 0 else tn / (tn + fn)
        negative_recall += 0 if tn + fp == 0 else tn / (tn + fp)

    metrics = [accuracy, positive_precision, positive_recall, negative_precision, negative_recall]

    metrics = [x / n_splits for x in metrics]

    return metrics


def calculate_metrics(clf, X, y, n_splits):
    metrics = cross_validation(clf, X, y, 1, n_splits)
    print('accuracy: ', metrics[0])
    print('Positive')
    print('precision: ', metrics[1])
    print('recall: ', metrics[2])
    print('Negative')
    print('precision: ', metrics[3])
    print('recall: ', metrics[4])
