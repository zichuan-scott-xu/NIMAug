import argparse
import os
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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--test_csv', type=str, help='test csv file')
parser.add_argument('--test_images', type=str, help='path to folder containing images')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--predictions', type=str, help='output file to store predictions')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
static_model = NIMA(models.vgg16(pretrained=True)).to(device)
static_model.load_state_dict(torch.load('checkpoint/epoch-85.pth'))
static_model.eval()

base_model = models.mobilenet_v3_large(pretrained=True)
sal_model = models.mobilenet_v3_small(pretrained=True)
model = NIMAug(base_model, sal_model)
model = model.to(device)

try:
    model.load_state_dict(torch.load(args.model))
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

test_df = pd.read_csv(args.test_csv, header=None)
test_imgs = test_df[0]
pbar = tqdm(total=len(test_imgs))

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
        out = model(imt, sal, hist, hog)
    out = out.view(10, 1)

    pred_scores = out
    mean, std, median, mode = 0.0, 0.0, 0, 0
    cumulative = 0.0
    cumulative_prev = 0.0
    e_biggest = 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
        cumulative += e
        if cumulative_prev < 0.5 and cumulative >= 0.5:
            median = j
        cumulative_prev += e
        if e > e_biggest:
            e_biggest = e
            mode = j 
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5
    gt = test_df[test_df[0] == img].to_numpy()[:, 1:].reshape(10, 1)
    gt_scores = gt
    gt_mean, gt_std, gt_median, gt_mode = 0.0, 0.0, 0, 0
    cumulative = 0
    cumulative_prev = 0
    e_biggest = 0.0
    for l, e in enumerate(gt, 1):
        gt_mean += l * e
        cumulative += e
        if cumulative_prev < 0.5 and cumulative >= 0.5:
            gt_median = l
        cumulative_prev += e
        if e > e_biggest:
            e_biggest = e
            gt_mode = l 
    for k, e in enumerate(gt, 1):
        gt_std += e * (k - gt_mean) ** 2
    gt_std = gt_std ** 0.5
    # print(str(img) + ' mean: %.3f | std: %.3f | GT: %.3f' % (mean, std, gt_mean))
    if not os.path.exists(args.predictions):
        os.makedirs(args.predictions)

    with open(os.path.join(args.predictions, 'pred.txt'), 'a') as f:
	# image name, 10 pred scores, mean, std, median, mode, 10 actual scores, mean, std, median, mode
        f.write(str(img) + ' %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d\n' % (pred_scores[0], pred_scores[1], pred_scores[2], pred_scores[3], pred_scores[4], pred_scores[5], pred_scores[6], pred_scores[7], pred_scores[8], pred_scores[9], mean, std, median, mode, gt_scores[0], gt_scores[1], gt_scores[2], gt_scores[3], gt_scores[4], gt_scores[5], gt_scores[6], gt_scores[7], gt_scores[8], gt_scores[9], gt_mean, gt_std, gt_median, gt_mode))

    pbar.update()
