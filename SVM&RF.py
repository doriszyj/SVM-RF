import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import math

#Set definitions for SVM
def svcClassify(trainData,trainLabel,testData):
    svcClf = svm.SVC(C=1.0, kernel='linear', gamma=0.0001,decision_function_shape='ovo', degree=3)
    svcClf.fit(trainData, trainLabel)
    testLable = svcClf.predict(testData)
    #saveResult(testLable,'sklearn_SVC_C=5.0_Result.csv')
    return testLable

#Set definitions for random forest
def RFClassify(trainData,trainLabel,testData):
    rfClf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    rfClf.fit_transform(trainData, trainLabel)
    testlabel=rfClf.predict(testData)
    return testlabel

#Test prints - uncomment for test
#print(rent_boston.data.shape)
#print(rent_boston.target)
#print(rent_boston.data)

#Load data set
rent_boston = datasets.load_boston()
#Set size of training set to learn from
n = 400

#Set training set
trainData = np.array(rent_boston.data[:n])
trainlabel = rent_boston.target[:n]

#Test set
testData = rent_boston.data[n+1:]
label_exact = rent_boston.target[n+1:]

#Convert to integer for analysis (to be converted back later)
n = 0
for i in trainlabel:
    i = i * 10
    trainlabel[n] = i
    n += 1

#Make SVM prediction and convert back to original decimal
SVM_label_predict = svcClassify(trainData, trainlabel, testData)/10 #est_median_value
#Calculate mean absolute error
error_SVM = metrics.mean_absolute_error(label_exact, SVM_label_predict)

#Make RF prediction and convert back to original decimal
RF_label_predict = RFClassify(trainData, trainlabel, testData)/10 #est_median_value
#Calculate mean absolute error
error_RF = metrics.mean_absolute_error(label_exact, RF_label_predict)

"""
error = 0
for i in range(105):
    error += pow((label_predict[i] - label_exact[i]),2)
error = math.sqrt(error)
"""

print "SVM predict label:", SVM_label_predict
print "SVM error is:" , error_SVM
print "RF predict label:", RF_label_predict
print "RF error is:" , error_RF

#Alternative printing - list form
# print "SVM predict label:"
# m=0
# for i in SVM_label_predict:
#     print SVM_label_predict[m]
#     m += 1
# print "SVM error is:" , error_SVM
#
# print "RF predict label:"
# l=0
# for i in RF_label_predict:
#     print RF_label_predict[l]
#     l += 1
# print "RF error is:" , error_RF
#
# print "Label exact:"
# k=0
# for i in label_exact:
#     print label_exact[k]
#     k += 1

plt.figure(figsize=(9,5),dpi=80)
plt.scatter(range(505-n), SVM_label_predict, label="SVM_predict_value", c="red", marker="x", s=30)
plt.scatter(range(505-n), RF_label_predict, label="RF_predict_value", c="blue", marker="+", s=30)
plt.scatter(range(505-n), label_exact, label="exact_value", c="green", marker="o", s=30)
plt.xlabel("Instance")
plt.ylabel("Median Value of Owner-occupied Homes in $1000's (1993)")
plt.title("Comparison of the Exact Value with SVM & RF prediction")
plt.legend()
plt.show()