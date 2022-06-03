import torch
import torch.nn as nn
import torch.nn.functional as F

class NIMAug(nn.Module):
    def __init__(self, base_model, sal_model, num_classes=10):
        super(NIMAug, self).__init__()
        self.base_net = base_model.features
        self.saliency_net = sal_model.features
        self.num_image_features = 960 * 7 * 7
        self.num_saliency_features = 576 * 7 * 7
        self.num_hog_features = 1764 
        self.num_hist_features = 1024

        self.num_fc_in = self.num_image_features // (7 * 7) * (3 * 3) + self.num_saliency_features // (7 * 7) * (3 * 3) + self.num_hog_features + self.num_hist_features

        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=self.num_fc_in, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.75),
            self.fc2,
            nn.Softmax()
        )

        
    def forward(self, x, sal, hist, hog):
        img_hidden = self.base_net(x)
        img_hidden = self.pool1(img_hidden)
        img_hidden = img_hidden.view(img_hidden.size(0), -1)

        sal_hidden = self.saliency_net(sal)
        sal_hidden = self.pool2(sal_hidden)
        sal_hidden = sal_hidden.view(sal_hidden.size(0), -1)

        hog = hog.view(hog.size(0), -1)
        hist = hist.view(hist.size(0), -1)

        x_aug = torch.cat([img_hidden, sal_hidden, hist, hog], dim=1)

        out = self.classifier(x_aug)
        return out
    
    def get_last_layer_feature(self, x, sal, hist, hog):
        img_hidden = self.base_net(x)
        img_hidden = self.pool1(img_hidden)
        img_hidden = img_hidden.view(img_hidden.size(0), -1)
        sal_hidden = self.saliency_net(sal)
        sal_hidden = self.pool2(sal_hidden)
        sal_hidden = sal_hidden.view(sal_hidden.size(0), -1)
        hog = hog.view(hog.size(0), -1)
        hist = hist.view(hist.size(0), -1)
        x_aug = torch.cat([img_hidden, sal_hidden, hist, hog], dim=1)
        classifier_no_softmax = nn.Sequential(
            nn.Dropout(p=0.75),
            self.fc1,
        )
        return classifier_no_softmax(x_aug)


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size
