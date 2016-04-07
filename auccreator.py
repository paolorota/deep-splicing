from sklearn import metrics
import matplotlib.pyplot as plt

def getAUC(y, pred, doShow=False, saveas=None):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    a = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    if saveas is not None:
        plt.savefig(saveas)
    if doShow:
        plt.show()
    return a, fpr, tpr
