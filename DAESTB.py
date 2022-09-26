import numpy as np
from matplotlib import pyplot
from numpy import interp
from dataprocessing import prepare_data, prepare_data2, preprocess_labels, calculate_performace
from DeepAE import DeepAE, DeepAE2, transfer_label_from_prob
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import time

def DAESTB():
    H, label = prepare_data()
    # H, label = prepare_data2()
    y, encoder = preprocess_labels(label)
    num = np.arange(len(y))
    y = y[num]

    encoder, H_data = DeepAE(H)
    # encoder, H_data = DeepAE2(H)

    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    num_cross = 5
    # num_cross = 10
    aucs = []
    all_performance = []

    tpr_list = []

    for fold in range(num_cross):
        train = np.array([x for i, x in enumerate(H_data) if i % num_cross != fold])
        test = np.array([x for i, x in enumerate(H_data) if i % num_cross == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross == fold])

        real_labels = []
        for val in test_label:
            if val[0] == 1:  # tuples in array, val[0]- first element of tuple
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_train = train
        prefilter_test = test

        clf = XGBClassifier(n_estimators=100, learning_rate=0.3)
        clf.fit(prefilter_train, train_label_new)  # Training
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        proba = transfer_label_from_prob(ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), proba,
                                                                                       real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print("DAESTB:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])

        t = t + 1  # AUC fold number

        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
        mean_tpr[0] = 0.0

        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC curve: 5-Fold CV')
        pyplot.legend()

    mean_tpr /= num_cross
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    np.savetxt('XGBoost_mean_fpr.csv', mean_fpr, delimiter=',')
    np.savetxt('XGBoost_mean_tpr.csv', mean_tpr, delimiter=',')

    # pyplot.plot(mean_fpr, mean_tpr, linewidth=1, alpha=0.8, label='DAESTB(AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc))
    pyplot.plot(mean_fpr, mean_tpr, linewidth=1, alpha=0.8,
                label='Mean ROC(AUC = %0.4f)' % mean_auc)

    pyplot.legend()

    print('std_auc=', std_auc)
    plt.savefig('5-fold CV DAESTB(AUC = %0.4f).png' % mean_auc, dpi=300)

    pyplot.show()
    print('*******AUTO-STB*****')
    print('mean performance of XGB using raw feature')
    print(np.mean(np.array(all_performance), axis=0))
    Mean_Result = np.mean(np.array(all_performance), axis=0)
    print('---' * 20)
    print('Mean-Accuracy=', Mean_Result[0], '\n Mean-precision=', Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=', Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6], '\n' 'Mean_F1=', Mean_Result[7])
    print('---' * 20)


if __name__=="__main__":
    time_start = time.time()

    DAESTB()

    time_end = time.time()
    print('time cost', time_end - time_start, 's')