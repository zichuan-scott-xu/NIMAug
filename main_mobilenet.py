import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import AVADataset

from model.model import *
from model.model_mobilenet import *

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

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


    #  Load old model 
    static_model = NIMA(models.vgg16(pretrained=True)).to(device)
    static_model.load_state_dict(torch.load('checkpoint/epoch-85.pth'))


    base_model = models.mobilenet_v3_large(pretrained=True)


    static_model.eval()

    sal_model = models.mobilenet_v3_small(pretrained=True)
    model = NIMAug(base_model, sal_model)
    model = model.to(device)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pth' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pth' % config.warm_start_epoch)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.base_net.parameters(), 'lr': conv_base_lr},
        {'params': model.saliency_net.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
    )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.img_path, transform=train_transform, sal_model=static_model)
        valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.img_path, transform=val_transform, sal_model=static_model)

        if config.use_subset:
            trainset = torch.utils.data.Subset(trainset, list(range(0, 500)))
            valset = torch.utils.data.Subset(valset, list(range(0, 500)))

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        last_check_epoch = 0
        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                sal = data['sal'].to(device)
                hist = data['hist'].to(device)
                hog = data['hog'].to(device)

                outputs = model(images, sal, hist, hog)
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()
                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))
                writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // config.train_batch_size + 1))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))
            with open(os.path.join(config.ckpt_path, 'train_log.txt'), 'a') as f:
                f.write('Epoch %d mean training EMD loss: %.4f.\n' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if config.decay:
                if (epoch + 1) % 10 == 0:
                    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    optimizer = optim.SGD([
                        {'params': model.base_net.parameters(), 'lr': conv_base_lr},
                        {'params': model.saliency_net.parameters(), 'lr': conv_base_lr},
                        {'params': model.classifier.parameters(), 'lr': dense_lr}],
                        momentum=0.9
                    )

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()

                sal = data['sal'].to(device)
                with torch.no_grad():
                    hist = data['hist'].to(device)
                    hog = data['hog'].to(device)
                    outputs = model(images, sal, hist, hog)

                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)
            print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
            with open(os.path.join(config.ckpt_path, 'train_log.txt'), 'a') as f:
                f.write('Epoch %d completed. Mean EMD loss on val set: %.4f.\n' % (epoch + 1, avg_val_loss))

            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
                old_checkpoint_path = os.path.join(config.ckpt_path, 'epoch-%d.pth' % (last_check_epoch))
                if os.path.exists(old_checkpoint_path):
                    print(epoch + 1)
                    os.remove(old_checkpoint_path)
                last_check_epoch = epoch + 1
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform, sal_model=static_model)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            images = data['image'].to(device)
            sal = data['sal'].to(device)
            hist = data['hist'].to(device)
            hog = data['hog'].to(device)
            outputs = model(images, sal, hist, hog)
            output = output.view(10, 1)

            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--img_path', type=str, default='./data/images')
    parser.add_argument('--train_csv_file', type=str, default='./data/train_labels.csv')
    parser.add_argument('--val_csv_file', type=str, default='./data/val_labels.csv')
    parser.add_argument('--test_csv_file', type=str, default='./data/test_labels.csv')

    # training parameters
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--dense_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--use_subset', type=bool, default=False)


    config = parser.parse_args()

    main(config)

