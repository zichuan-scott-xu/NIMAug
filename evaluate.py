import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str, help='path to prediction txt')
args = parser.parse_args()
filepath = args.pred

# read text file into pandas DataFrame
df = pd.read_csv(filepath, sep=" ", names=["File Name", "Pred_score 1", "Pred_score 2", "Pred_score 3", "Pred_score 4", "Pred_score 5", "Pred_score 6", "Pred_score 7", "Pred_score 8", "Pred_score 9", "Pred_score 10", "Pred_mean", "Pred_std", "Pred_median", "Pred_mode", "gt_score 1", "gt_score 2", "gt_score 3", "gt_score 4", "gt_score 5", "gt_score 6", "gt_score 7", "gt_score 8", "gt_score 9", "gt_score 10", "gt_mean", "gt_std", "gt_median", "gt_mode"])

pred_class = [df["Pred_mean"] > 5][0]
gt_class = [df["gt_mean"] > 5][0]
pred_class = 1 * pred_class.to_numpy()
gt_class = 1 * gt_class.to_numpy()
accuracy = accuracy_score(gt_class, pred_class)
print()
print("----------Model Evaluation----------")
print()
print("Binary Classification Accuracy is " + "%.5f" %(accuracy))
print("Binary Classification Precision is "+ "%.5f" %(precision_score(gt_class, pred_class)))
print("Binary Classification Recall is " + "%.5f" %(recall_score(gt_class, pred_class)))
print("Binary Classification F1 Score is " + "%.5f\n" %(f1_score(gt_class, pred_class)))

print("Linear Correlation Coefficient of Mean is " + "%.5f" %(np.corrcoef(df["Pred_mean"], df["gt_mean"])[0,1]))
print("Spearman's Rank Correlation Coefficient of Mean is " + "%.5f" %(spearmanr(df["Pred_mean"], df["gt_mean"])[0]))
print("Linear Correlation Coefficient of Std is " + "%.5f" %(np.corrcoef(df["Pred_std"], df["gt_std"])[0,1]))
print("Spearman's Rank Correlation Coefficient of Std is " + "%.5f" %(spearmanr(df["Pred_std"], df["gt_std"])[0]))
from model.model import *
p = df[["Pred_score 1", "Pred_score 2", "Pred_score 3", "Pred_score 4", "Pred_score 5", "Pred_score 6", "Pred_score 7", "Pred_score 8", "Pred_score 9", "Pred_score 10"]]
q = df[["gt_score 1", "gt_score 2", "gt_score 3", "gt_score 4", "gt_score 5", "gt_score 6", "gt_score 7", "gt_score 8", "gt_score 9", "gt_score 10"]]
p = torch.from_numpy(p.to_numpy())
q = torch.from_numpy(q.to_numpy())
print("EMD Loss is "+"%.5f" %(emd_loss(p,q)))
print()

mean = df["Pred_mean"].to_numpy()
gt_mean = df["gt_mean"].to_numpy()

# new code
diff = np.abs(mean - gt_mean)
index1 = np.argpartition(diff, 5)
index2 = np.argpartition(diff, -5)
index3 = np.argpartition(mean, -5)
index4 = np.argpartition(mean, 5)  
# end of new code

mean = np.rint(mean)
gt_mean = np.rint(gt_mean)
mean_class = mean > 5
gt_mean_class = gt_mean > 5

print("----------10-Class Classification----------")
print("Binary Rounded Mean Accuracy is " + "%.5f" %(accuracy_score(mean_class, gt_mean_class)))
print("Mean Accuracy is " + "%.5f" %(accuracy_score(mean, gt_mean)))
print("Median Accuracy is " + "%.5f" %(accuracy_score(df["Pred_median"], df["gt_median"])))
print("Mode Accuracy is " + "%.5f" %(accuracy_score(df["Pred_mode"], df["gt_mode"])))
print("Image with largest mean score is", int(df.iloc[np.argmax(df["Pred_mean"])][0]))
print("Image with lowest mean score is", int(df.iloc[np.argmin(df["Pred_mean"])][0]))
print()

# New Code
print("5 images with largest mean score are:", end = " ")
for i in range(1, 5):
    print(int(df.iloc[index3[-i]][0]), end = "; ")
print(int(df.iloc[index3[-5]][0]))

print("5 images with lowest mean score are:", end = " ")
for i in range(4):
    print(int(df.iloc[index4[i]][0]), end = "; ")
print(int(df.iloc[index4[4]][0]))

print("5 images with best prediction are:", end = " ")
for i in range(4):
    print(int(df.iloc[index1[i]][0]), end = "; ")
print(int(df.iloc[index1[4]][0]))

print("5 images with worst prediction are:", end = " ")
for i in range(1, 5):
    print(int(df.iloc[index2[-i]][0]), end = "; ")
print(int(df.iloc[index2[-5]][0]))
# End of New Code

global_mean = np.mean(df["Pred_mean"].to_numpy())
print("Mean score over all images is "+"%.5f" %(global_mean))
array = np.asarray(df["Pred_mean"])
idx = np.argpartition(np.abs(array-global_mean), 5)
print("5 images closest to the mean score is:", end = " ")
for i in range(4):
    print(int(df.iloc[idx[i]][0]), end = "; ")
print(int(df.iloc[idx[4]][0]))
print()
