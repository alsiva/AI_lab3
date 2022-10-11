import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, RocCurveDisplay, \
    PrecisionRecallDisplay
import matplotlib.pyplot as plt


def draw_plt(predict_arr, expect_arr):
    y_true = np.array([0 if x == 'p' else 1 for x in predict_arr])
    y_pred = np.array([0 if x == 'p' else 1 for x in expect_arr])

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    display_roc_auc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display_roc_auc.plot()

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    display_roc_pr = PrecisionRecallDisplay(precision=precision, recall=recall)
    display_roc_pr.plot()

    plt.show()
