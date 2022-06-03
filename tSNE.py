import argparse
import os
from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import *
from model.model_mobilenet import *

from model.hist_utils import histogram
from model.hog_utils import get_hog_feature
from model.sal_utils import saliency

from sklearn.manifold import TSNE


# python -W ignore tSNE.py --model checkpoint_mobilenet/epoch-28.pth --test_csv ~/NIA/AVA_dataset/labels/test_labels.csv --test_images ~/NIA/AVA_dataset/images/images --predictions predictions/             

"""
Params:
    features: numpy array with dimension (N, D) where N is the number of samples with D dimensions
    labels: numpy array with dimension (N, 1)
"""
def plot_TSNE(features, labels, save=False):
    neg = labels == 0
    pos = labels == 1
    tsne_embedding = TSNE(n_components=2, init='random').fit_transform(features)
    neg_x = tsne_embedding[neg, 0] 
    neg_y = tsne_embedding[neg, 1]
    pos_x = tsne_embedding[pos, 0] 
    pos_y = tsne_embedding[pos, 1] 
    plt.figure()
    plt.scatter(neg_x, neg_y, color='red', label="negative")
    plt.scatter(pos_x, pos_y, color='green', label="positive")
    plt.legend()
    if save:
        plt.savefig('tsne.png')
    plt.show()


def get_mean_score(y):
    return sum([(i+1) * y[i] for i in range(10)])


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--test_csv', type=str, help='test csv file')
parser.add_argument('--test_images', type=str, help='path to folder containing images')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--predictions', type=str, help='output file to store predictions')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
static_model = NIMA(models.vgg16(pretrained=True)).to(device)
static_model.load_state_dict(torch.load('checkpoint/epoch-85.pth', map_location=device))
static_model.eval()

base_model = models.mobilenet_v3_large(pretrained=True)
sal_model = models.mobilenet_v3_small(pretrained=True)
model = NIMAug(base_model, sal_model)
model = model.to(device)

try:
    model.load_state_dict(torch.load(args.model, map_location=device))
    print('successfully loaded model')
except:
    raise


seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

N_EVALUATE = 50
N_SAMPLE = 2000
pos_count = 0
neg_count = 0

test_df = pd.read_csv(args.test_csv, header=None).sample(n=N_SAMPLE)
test_df.reset_index(drop=True, inplace=True)
test_imgs = test_df[0]
pbar = tqdm(total=N_EVALUATE * 2)


Xs = np.zeros((len(test_imgs), 256))
ys = np.zeros(len(test_imgs),)

for i, img in enumerate(test_imgs):
    im = Image.open(os.path.join(args.test_images, str(img) + '.jpg'))
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)

# Modified
    sal = saliency(imt, static_model).to(device)
    with torch.no_grad():
        hist = histogram(imt).to(device)
        hog = get_hog_feature(imt).to(device)
        features = model.get_last_layer_feature(imt, sal, hist, hog)
        gt = test_df[test_df[0] == img].to_numpy()[:, 1:].reshape(10, 1)
        
        if pos_count < N_EVALUATE and get_mean_score(gt) > 5:
            Xs[pos_count + neg_count] = features.cpu().detach().numpy()
            ys[pos_count + neg_count] = 1
            pos_count += 1
            pbar.update()
        elif neg_count < N_EVALUATE and get_mean_score(gt) <= 5:
            Xs[pos_count + neg_count] = features.cpu().detach().numpy()
            ys[pos_count + neg_count] = 0
            neg_count += 1
            pbar.update()
        elif pos_count >= N_EVALUATE and neg_count >= N_EVALUATE:
            break

plot_TSNE(Xs, ys, save=True)