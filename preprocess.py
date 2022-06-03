import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

from model.hist_utils import histogram
from model.hog_utils import get_hog_feature
from model.sal_utils import saliency
from dataset.dataset import AVADataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from model.model import *

train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load("checkpoint/epoch-85.pth"))
# Set the model to run on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model on Eval Mode
model.eval()

trainset = AVADataset(csv_file="AVA_dataset/labels/train_labels.csv", root_dir="AVA_dataset/images/images", transform=train_transform)
valset = AVADataset(csv_file="AVA_dataset/labels/val_labels.csv", root_dir="AVA_dataset/images/images", transform=val_transform)
testset = AVADataset(csv_file="AVA_dataset/labels/test_labels.csv", root_dir="AVA_dataset/images/images", transform=val_transform)

for i in range(198537, len(trainset)):
    im_id = trainset[i]["img_id"]
    image = trainset[i]["image"].cuda()
    image = image.unsqueeze(0)
    sal = saliency(image, model)
    sal = sal.squeeze(0)
    torch.save(sal, "AVA_dataset/images/sal/" + im_id[26:-4] + "_sal.pt")
    hog = get_hog_feature(image)
    hog = hog.squeeze(0)
    torch.save(hog, "AVA_dataset/images/hog/" + im_id[26:-4] + "_hog.pt")
    hist = histogram(image)
    hist = hist.squeeze(0)
    torch.save(hist, "AVA_dataset/images/hist/" + im_id[26:-4] + "_hist.pt")
    if (i+1) % 5000 == 0:
        print("Finished the first", i, "images in the training set.")
    
for i in range(len(valset)):
    im_id = valset[i]["img_id"]
    image = valset[i]["image"].cuda()
    image = image.unsqueeze(0)
    sal = saliency(image, model)
    sal = sal.squeeze(0)
    torch.save(sal, "AVA_dataset/images/sal/" + im_id[26:-4] + "_sal.pt")
    hog = get_hog_feature(image)
    hog = hog.squeeze(0)
    torch.save(hog, "AVA_dataset/images/hog/" + im_id[26:-4] + "_hog.pt")
    hist = histogram(image)
    hist = hist.squeeze(0)
    torch.save(hist, "AVA_dataset/images/hist/" + im_id[26:-4] + "_hist.pt")    
    if (i+1) % 5000 == 0:
        print("Finished the first", i, "images in the validation set.")
    
for i in range(len(testset)):
    im_id = testset[i]["img_id"]
    image = testset[i]["image"].cuda()
    image = image.unsqueeze(0)
    sal = saliency(image, model)
    sal = sal.squeeze(0)
    torch.save(sal, "AVA_dataset/images/sal/" + im_id[26:-4] + "_sal.pt")
    hog = get_hog_feature(image)
    hog = hog.squeeze(0)
    torch.save(hog, "AVA_dataset/images/hog/" + im_id[26:-4] + "_hog.pt")
    hist = histogram(image)
    hist = hist.squeeze(0)
    torch.save(hist, "AVA_dataset/images/hist/" + im_id[26:-4] + "_hist.pt")
    if (i+1) % 5000 == 0:
        print("Finished the first", i, "images in the testing set.")
