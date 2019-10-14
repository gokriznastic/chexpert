import sys

# import cv2
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torchvision import transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

###################### loading data ######################

# def load_and_resize_img(path):
#     """
#     Load and convert the full resolution images on CodaLab to
#     low resolution used in the small dataset.
#     """
#     img = cv2.imread(path, 0)

#     size = img.shape
#     max_dim = max(size)
#     max_ind = size.index(max_dim)

#     if max_ind == 1:
#         # width fixed at 320
#         wpercent = (320 / float(size[0]))
#         hsize = int((size[1] * wpercent))
#         new_size = (hsize, 320)

#     else:
#         # height fixed at 320
#         hpercent = (320 / float(size[1]))
#         wsize = int((size[0] * hpercent))
#         new_size = (320, wsize)

#     resized_img = cv2.resize(img, new_size)

#     return resized_img


def ImageTensor(image_name):
    image = Image.open(image_name).convert('RGB')

    transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    image = transform(image)

    return image_name, image

input_csv_filename = sys.argv[1]
prediction_csv_filename = sys.argv[2]

df = pd.read_csv(input_csv_filename)
image_list = np.asarray(df)

loader = [ImageTensor(image_name[0]) for image_name in image_list]


###################### model ######################

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
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

        return torch.sigmoid(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DenseNet121(5)
model = torch.nn.DataParallel(model)
model = model.to(device)

###################### testing ######################

def test(test_loader, model, checkpoint, device):
        cudnn.benchmark = True

        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        pred = []

        model.eval()
        with torch.no_grad():
            for name, data in test_loader:
                pred_ = []

                data = data.view(-1, data.size(0), data.size(1), data.size(2)).to(device)

                out = model(data).cpu().numpy()
                # print(out)

                pred_.append(name)
                for prob in out[0]:
                    pred_.append(prob)

                pred.append(pred_)

        return pred

class_names = ['Study', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

pred = test(loader, model, 'src/10-10-2019-19:13:51_model.pt', device)

pred_df = pd.DataFrame(pred, columns=class_names)

image_names = pred_df['Study']
temp = []
for image_name in image_names:
    image_name = '/'.join(image_name.split('/')[:-1])
    temp.append(image_name)
pred_df['Study'] = temp

def merge_studies(df):
    dic = {'Atelectasis': 'max', 'Cardiomegaly': 'max', 'Consolidation': 'max', 'Edema': 'max', 'Pleural Effusion': 'max'}
    df_new = df.groupby(df['Study'], as_index=False).aggregate(dic).reindex(columns=df.columns)

    return df_new

pred_df = merge_studies(pred_df)

pd.DataFrame.to_csv(pred_df, prediction_csv_filename, index=False)