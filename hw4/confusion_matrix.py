from sklearn import metrics

def confution_matrix(y, pred):
    return metrics.confusion_matrix(y, pred)
