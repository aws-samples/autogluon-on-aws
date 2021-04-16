import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, average_precision_score,
                            precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, auc)

def plot_roc_curve(y_true, y_score, is_single_fig=False):
    """
    Plot ROC Curve and show AUROC score
    """    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('AUROC = {:.4f}'.format(roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('TPR(True Positive Rate)')
    plt.xlabel('FPR(False Positive Rate)')
    if is_single_fig:
        plt.show()
    
def plot_pr_curve(y_true, y_score, is_single_fig=False):
    """
    Plot Precision Recall Curve and show AUPRC score
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)
    plt.title('AUPRC = {:.4f}'.format(avg_prec))
    plt.step(rec, prec, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
    plt.plot(rec, prec, 'b')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if is_single_fig:
        plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['0','1'], is_single_fig=False):
    """
    Plot Confusion matrix
    """    
    y_pred = np.where(y_score >= thresh, 1, 0)
    print("confusion matrix (cutoff={})".format(thresh))
    print(classification_report(y_true, y_pred, target_names=class_labels))
    conf_mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if is_single_fig:
        plt.show()
        
def plot_all(y_true, y_score, thresh=0.5):

    if y_true.dtype == 'object':
        y_true, uniques = pd.factorize(y_true)    
    
    fig = plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plot_roc_curve(y_true, y_score)
    plt.subplot(1,3,2)    
    plot_pr_curve(y_true, y_score)
    plt.subplot(1,3,3)    
    plot_conf_mtx(y_true, y_score, thresh)         