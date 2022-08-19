from sklearn.metrics import roc_auc_score,recall_score,f1_score
import numpy as np

def get_metrics(y_true,y_pred,y_score=None):
    y_true = np.reshape(y_true,(-1,))
    y_pred = np.reshape(y_pred,(-1,))
    y_score = np.reshape(y_score,(-1,))
    y_pred = np.round(y_pred)
    f1_s = f1_score(y_true=y_true,y_pred=y_pred)
    recall = recall_score(y_true=y_true,y_pred=y_pred)
    if y_score is not None:
        auc = roc_auc_score(y_true=y_true,y_score=y_score)
    else:
        auc = roc_auc_score(y_true=y_true,y_score=y_pred)
    fpr = get_fpr(y_true,y_pred=y_pred)
    return (recall,f1_s,auc,fpr)

def get_fpr(y_true,y_pred):
    fp = np.sum(np.multiply((1 - y_true), y_pred))
    tn = np.sum(np.multiply((1 - y_true), (1 - y_pred)))
    return fp / (fp + tn)