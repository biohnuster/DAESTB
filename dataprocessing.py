import numpy as np
from numpy import matlib as mb
from sklearn import preprocessing
from keras.utils import np_utils

# 异构网络的构建
# dataset1
def prepare_data():
    MM = np.loadtxt(r".\dataset\dataset1\miRNA similarity matrix.txt", delimiter="\t")
    SM = np.loadtxt(r".\dataset\dataset1\SM similarity matrix.txt", delimiter="\t")
    A = np.loadtxt(r".\dataset\dataset1\known miRNA-SM association matrix.txt", dtype=int, delimiter="\t")

    mm = np.repeat(MM, repeats=831, axis=0)

    sm = mb.repmat(SM, 541, 1)
    H = np.concatenate((sm, mm), axis=1)  # (449571,1372)

    label = A.reshape((449571, 1))

    return H, label
# dataset2
def prepare_data2():
    MM = np.loadtxt(r".\dataset\dataset2\miRNA similarity matrix2.txt", delimiter=",")
    SM = np.loadtxt(r".\dataset\dataset2\SM similarity matrix2.txt", delimiter=",")
    A = np.loadtxt(r".\dataset\dataset2\known miRNA-SM association matrix2.txt", delimiter=",")


    mm = np.repeat(MM, repeats=39, axis=0)

    sm = mb.repmat(SM, 286, 1)
    H = np.concatenate((sm, mm), axis=1)  # (11154,325)

    label = A.reshape((11154, 1))

    return H, label

# one-hot
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)  # one-hot
    return y, encoder

def calculate_performace(test_num, pred_y, labels): # pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
    return acc, precision, sensitivity, specificity, MCC, f1_score


def transfer_array_format(data):  # data=X  , X= all the miRNA features, disease features
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])  # contains miRNA features
        formated_matrix2.append(val[1])  # contains small molecule features

    return np.array(formated_matrix1), np.array(formated_matrix2)


# H, label = prepare_data()
# print(H.shape)
# print(label.shape)