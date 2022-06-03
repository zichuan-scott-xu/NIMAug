# This utility function calculates the saliency map of an image with respect
# to a pre-trained model. It returns the 224*224 saliency map. 

import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from model.model import *

def saliency(images, model):
    bs = images.shape[0]
    result = torch.zeros((bs, 3, 224, 224))
    for i in range(bs):
        image = images[i]
        image = image.reshape(1, 3, 224, 224)
        # Set the requires_grad_ to the image for retrieving gradients
        image.requires_grad_()
        # Retrieve output from the image
        output = model(image)

        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]

        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()
    
        # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
        saliency = saliency.reshape(224, 224)
        result[i, 0, :, :] = saliency
        result[i, 1, :, :] = saliency
        result[i, 2, :, :] = saliency
    return result
