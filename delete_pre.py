import os
import numpy as np
from numpy import genfromtxt
data = genfromtxt('AVA_dataset/labels/test_labels.csv', delimiter=',')
for i in range(data.shape[0]):
    filePath = "AVA_dataset/images/hist/"+str(int(data[i][0]))+"_hist.pt"
    if os.path.exists(filePath):
        os.remove(filePath)
