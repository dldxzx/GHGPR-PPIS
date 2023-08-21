# -*- encoding:utf8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_aupr(preds, labels):
    # Compute ROC curve and ROC area for each class
    p, r, _ = precision_recall_curve(labels, preds)
    aupr = auc(r, p)
    return aupr


def compute_mcc(preds, labels, threshold=0.5):
    # preds = preds.astype(np.float64)
    # labels = labels.astype(np.float64)
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels, preds)
    return mcc


def compute_performance(preds, labels):
    predictions_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        # print('===========================')
        # print(predictions)
        p = 0.0
        r = 0.0
        total = 0
        p_total = 0
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp

        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max <= f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    predictions_max = predictions

    return f_max, p_max, r_max, t_max, predictions_max




def compute_performance_self_res(preds, labels, res_self_pro):
    predictions_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    predictions = (preds > res_self_pro).astype(np.int32)
    # print('===========================')
    # print(predictions)
    p = 0.0
    r = 0.0
    total = 0
    p_total = 0
    tp = np.sum(predictions * labels)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp
    # if tp == 0 and fp == 0 and fn == 0:
    #     continue
    total += 1
    if tp != 0:
        p_total += 1
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        p += precision
        r += recall
    if total > 0 and p_total > 0:
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max <= f:
                f_max = f
                p_max = p
                r_max = r
                t_max = 0
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max














def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-12)
    MiR = TP / max(total_R, 1e-12)
    if TP == 0:
        MiF = 0
    else:
        MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF, total_P / N, total_R / N


def acc_score(output, label):
    acc = accuracy_score(label, output)

    return acc


# if __name__ == '__main__':
#     pass


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def calc(TN, FP, FN, TP):
    recall = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return recall, Precision, F1, ACC, MCC


# y_true = [0, 1, 0, 0, 1, 1, 1, 1, 1]
# y_pred = [0, 1, 1, 1, 1, 1, 1, 1, 1]
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# recall, pre, f1, acc, mcc = calc(tn, fp, fn, tp)


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]

    return binary_pred, best_threshold


def computeperformance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall, pre, f1, acc, mcc = calc(tn, fp, fn, tp)
    return recall, pre, f1, acc, mcc
