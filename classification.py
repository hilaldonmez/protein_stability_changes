import preprocessing as pr
import feature_extraction as fe
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support


def evaluate (clf, X, y):
    n_splits = 4
    kf = KFold(n_splits)
    kf.get_n_splits(X)
    
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_fscore = 0
    total_support = 0
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       clf.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       total_acc = total_acc + accuracy_score(y_test, y_pred)
       print(precision_recall_fscore_support(y_test, y_pred, average = None ,labels=[0, 1]) )
       precision, recall , f_score , support = precision_recall_fscore_support(y_test, y_pred, average = None ,labels=[0, 1])   
       total_precision = total_precision + precision
       total_recall = total_recall + recall
       total_fscore = total_fscore + f_score
       total_support = total_support + support
       
    mean_acc = total_acc / n_splits   
    mean_pre = total_precision / n_splits 
    mean_recall = total_recall / n_splits 
    mean_f = total_fscore / n_splits 
    mean_support = total_support / n_splits
    

    print("Avg accuracy : ", mean_acc)
    print("Avg precision : ", mean_pre)
    print("Avg recall : ", mean_recall)
    print("Avg f : ", mean_f)
    print("Avg support : ", mean_support)
    
    

#%%
y = np.array(pr.label)
X = np.array(fe.SO_vectors_original)    
clf = LinearSVC(random_state=0)
evaluate(clf, X, y) 

          