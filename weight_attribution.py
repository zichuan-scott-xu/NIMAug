import argparse
import os
from attr import attrib
from cv2 import HOGDescriptor_getDefaultPeopleDetector
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import sys

from model.model import *
from model.model_mobilenet import *

from model.hist_utils import histogram
from model.hog_utils import get_hog_feature
from model.sal_utils import saliency

from sklearn.manifold import TSNE
from captum.attr import IntegratedGradients

# sample usage: python -W ignore weight_attribution.py --model checkpoint_mobilenet/epoch-28.pth --test_csv ~/NIA/AVA_dataset/labels/test_labels.csv --test_images ~/NIA/AVA_dataset/images/images --predictions predictions/ 

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

N_EVALUATE = 1000
test_df = pd.read_csv(args.test_csv, header=None).sample(n=N_EVALUATE)
test_df.reset_index(drop=True, inplace=True)
test_imgs = test_df[0]
pbar = tqdm(total=len(test_imgs))


Xs = np.zeros((len(test_imgs), 256))
ys = np.zeros(len(test_imgs),)

max_attr = {'img':0, 'sal':0, 'hist':0, 'hog':0}

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

        ig = IntegratedGradients(model)
        model_in = (imt, sal, hist, hog)
        for y in range(10):
            imt_attr, sal_attr, hist_attr, hog_attr = ig.attribute(model_in, target=y)
            max_attr['img'] += torch.max(torch.abs(imt_attr))
            max_attr['sal'] += torch.max(torch.abs(sal_attr))
            max_attr['hist'] += torch.max(torch.abs(hist_attr))
            max_attr['hog'] += torch.max(torch.abs(hog_attr))

    pbar.update()

for attr_name in max_attr:
    print(f"Average max attribution of {attr_name}: {max_attr[attr_name] / 10 / len(test_imgs)}")
