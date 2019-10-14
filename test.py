import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics.ranking import roc_auc_score

import torch
import torch.backends.cudnn as cudnn
from dataloader import CheXpertDataset, getImagesLabels
from network import DenseNet121
from torch.utils.data import DataLoader
from utils import categorical
from viz import plot_ROC


def computeAUROC (pred, gt, num_classes):
    AUROC = []

    np_pred = pred.cpu().numpy()
    np_gt = gt.cpu().numpy()

    for i in range(num_classes):
        try:
            AUROC.append(roc_auc_score(np_gt[:, i], np_pred[:, i]))
        except ValueError:
            pass
    return AUROC

def test(test_loader, model, checkpoint, device, num_classes, class_names):
        cudnn.benchmark = True
        threshold = 0.45

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])

        gt = torch.FloatTensor().to(device)
        pred = torch.FloatTensor().to(device)


        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):

                target = target.to(device)
                gt = torch.cat((gt, target), 0).to(device)

                out = model(data)
                # print(out)

                # zeros = torch.zeros_like(target)
                # ones = torch.ones_like(target)
                # out = torch.where(out>threshold, ones, zeros)
                # out = torch.from_numpy(categorical(out.cpu().numpy())).float().to(device)
                pred = torch.cat((pred, out), 0)

        auroc_individual = computeAUROC(pred, gt, num_classes)
        auroc_mean = np.array(auroc_individual).mean()

        print ('\nmean AUROC:\t\t{}\n'.format(auroc_mean))

        for i in range(len(auroc_individual)):
            print (class_names[i], '\t\t', auroc_individual[i])

        return pred, gt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 5
class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

val_csv = 'CheXpert-v1.0-small/valid.csv'
test_images, test_labels = getImagesLabels(val_csv, 'ones')

print('Testing on {} images\n'.format(len(test_images)))

test_dataset = CheXpertDataset(test_images, test_labels, test=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,  num_workers=24, pin_memory=True)

model = DenseNet121(num_classes, test=True)
model = torch.nn.DataParallel(model)

pred, gt = test(test_loader, model, '10-10-2019-19:13:51_model.pt', device, num_classes, class_names) # U-ignore
plot_ROC(pred, gt, num_classes, class_names)

pred[pred>=0.5] = 1
pred[pred<0.5] = 0

print('\nAccuracy:\n')
for i in range(num_classes):
    acc = accuracy_score(gt.cpu()[:,i], pred.cpu()[:,i])
    print (class_names[i], '\t\t', acc)


# pred_zeros, gt_zeros = test(test_loader, model, '10-10-2019-15:33:21_model.pt', device, num_classes, class_names) # U-zeros
# pred_ones, gt_ones = test(test_loader, model, '10-10-2019-14:39:11_model.pt', device, num_classes, class_names) # U-ones

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 20
# fig_size[1] = 4
# plt.rcParams["figure.figsize"] = fig_size
# plt.tight_layout()

# for i in range(num_classes):
#     fpr0, tpr0, threshold0 = metrics.roc_curve(gt_zeros.cpu()[:,i], pred_zeros.cpu()[:,i])
#     roc_auc0 = metrics.auc(fpr0, tpr0)
#     fpr1, tpr1, threshold1 = metrics.roc_curve(gt_ones.cpu()[:,i], pred_ones.cpu()[:,i])
#     roc_auc1 = metrics.auc(fpr1, tpr1)

#     plt.subplot(1, 5, i+1)

#     plt.title('ROC for: ' + class_names[i])
#     plt.plot(fpr0, tpr0, label = 'U-zeros: AUC = %0.2f' % roc_auc0)
#     plt.plot(fpr1, tpr1, label = 'U-ones: AUC = %0.2f' % roc_auc1)

#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([-0.01, 1])
#     plt.ylim([0, 1.01])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')

# # mng = plt.get_current_fig_manager()
# # mng.resize(*mng.window.maxsize())

# plt.savefig("ROC.png", dpi=1000)
# plt.show()
