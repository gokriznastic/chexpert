import torch
import  torchvision
import torch.nn as nn
from torchsummary import summary

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, test=False):
        super(DenseNet121, self).__init__()
        self.test = test
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_ftrs//2),
            nn.BatchNorm1d(num_ftrs//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(num_ftrs//2, out_size),
        )

    def forward(self, x):
        x = self.densenet121(x)

        if self.test:
            return torch.sigmoid(x)
        else:
            return x

# model = DenseNet121(5)
# # print(model)
# print(summary(model.cuda(), (3, 320, 320)))
