import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics.ranking import roc_auc_score


def plot_ROC(pred, gt, num_classes, class_names):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()

    for i in range(num_classes):
        fpr, tpr, threshold = metrics.roc_curve(gt.cpu()[:,i], pred.cpu()[:,i])
        roc_auc = metrics.auc(fpr, tpr)

        plt.subplot(1, 5, i+1)

        plt.title('ROC for: ' + class_names[i])
        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    plt.savefig("ROC.png", dpi=1000)
    plt.show()
