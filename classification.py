import preprocessing as pr
import feature_extraction as fe
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from sklearn.model_selection import train_test_split


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
def get_result_label(file_name):
    result_label = []
    with open(file_name) as f:
        for line in f:
            result_label.append(line.split()[0])
    return result_label

def write_file(file_name, X, y):
    file = open(file_name, "w")
    temp = ""    
    for mut,label in zip(X,y):
        temp = temp + str(label)
        for i in range(len(mut)):
            temp = temp + " "+ str(i+1) + ":" + str(mut[i]) + " "
        temp = temp + "\n"
    
    file.write(temp)

# cross validation is necessary, but not implemented yet
def SVM_classification(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    write_file("train.txt" , X_train , y_train)
    write_file("test.txt" , X_test , y_test)
    os.system("./svm_multiclass_learn -c 5.0 -t 2 -g 0.1 train.txt model_file")
    os.system("./svm_multiclass_classify test.txt model_file result.txt")
    result_label = get_result_label("result.txt")            
    result_label = list(map(int, result_label))
    precision, recall , f_score , support = precision_recall_fscore_support(result_label, y_test, average = None ,labels=[0, 1])   
    print("Precision: ", precision, "Recall : ", recall , "F Score: ", f_score , "Support : ", support)

#%%
y = np.array(pr.label)
X = np.array(fe.SO_vectors_original)    
SVM_classification(X,y)


       

          