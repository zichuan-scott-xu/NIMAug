# This utility function calculates the RGB channel histogram and 
# the grayscale histogram.

import torch
import torchvision.transforms as transforms

def histogram(tensors):
    bs = tensors.shape[0]
    result = torch.zeros((bs, 1024))
    for i in range(bs):
        tensor = tensors[i]
        red = tensor[0] * 0.229 + 0.485
        green = tensor[1] * 0.224 + 0.456
        blue = tensor[2] * 0.225 + 0.406
        grayscale = red * 0.229 + green * 0.587 + blue * 0.144
        hist_1 = torch.histc(red, bins=256, min=0, max=1, out=None)
        hist_2 = torch.histc(green, bins=256, min=0, max=1, out=None)
        hist_3 = torch.histc(blue, bins=256, min=0, max=1, out=None)
        hist_4 = torch.histc(grayscale, bins=256, min=0, max=1, out=None)
        out = torch.cat((hist_1, hist_2, hist_3, hist_4))
        out = out / 50176
        result[i] = out
    return result
