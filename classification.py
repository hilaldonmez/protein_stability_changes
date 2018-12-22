from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# %%
def get_result_label(file_name):
    result_label = []
    with open(file_name) as f:
        for line in f:
            result_label.append(line.split()[0])
    return result_label


def write_file(file_name, X, y):
    file = open(file_name, "w")
    temp = ""
    for mut, label in zip(X, y):
        temp = temp + str(label)
        for i in range(len(mut)):
            temp = temp + " " + str(i + 1) + ":" + str(mut[i]) + " "
        temp = temp + "\n"

    file.write(temp)


def train_random_forest(X, y, n_splits):
    clf = RandomForestClassifier(n_estimators=100)
    calculate_metrics(clf, X, y, n_splits)


def train_svm(X, y, n_splits):
    clf = svm.SVC(C=5, gamma=0.1, kernel='rbf')
    calculate_metrics(clf, X, y, n_splits)


def cross_validation(clf, X, y, label, n_splits):
    kf = KFold(n_splits)
    kf.get_n_splits(X)
    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
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

        mean_accuracy += (tp + tn) / (tp + fn + fp + tn)
        mean_precision += tp / (tp + fp)
        mean_recall += tp / (tp + fn)

    return mean_accuracy / n_splits, mean_precision / n_splits, mean_recall / n_splits


def calculate_metrics(clf, X, y, n_splits):
    mean_accuracy, mean_precision, mean_recall = cross_validation(clf, X, y, 1, n_splits)
    print('Label 1')
    print('accuracy: ', mean_accuracy)
    print('precision: ', mean_precision)
    print('recall: ', mean_recall)

    mean_accuracy, mean_precision, mean_recall = cross_validation(clf, X, y, 0, n_splits)
    print('Label 0')
    print('accuracy: ', mean_accuracy)
    print('precision: ', mean_precision)
    print('recall: ', mean_recall)
