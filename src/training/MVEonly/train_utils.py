import torch
import gc
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn.metrics import r2_score
from astropy.visualization import make_lupton_rgb
from sklearn.manifold import Isomap
import os
import json


# Load data function
def create_dataloader(img_path, metadata_path, batch_size):
    '''
    Creates dataloader for training, reserving the last 10% images for validation/testing
    '''
    data = np.load(img_path).squeeze()
    length = len(data)
    data_train = torch.tensor(data[:int(.7*length)]) # 70% train
    data_test = torch.tensor(data[int(.7*length):int(.9*length)]) # 20% test
    data_val = torch.tensor(data[int(.9*length):]) # 10% validation

    metadata = pd.read_csv(metadata_path)
    labels = metadata['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g'].tolist()
    labels_train = torch.tensor(labels[:int(.7*length)])
    labels_test = torch.tensor(labels[int(.7*length):int(.9*length)])
    labels_val = torch.tensor(labels[int(.9*length):])

    data_train.cuda()
    data_test.cuda()
    data_val.cuda()
    labels_train.cuda()
    labels_test.cuda()
    labels_val.cuda()

    train_dataset = TensorDataset(data_train, labels_train)
    test_dataset = TensorDataset(data_test, labels_test)
    val_dataset = TensorDataset(data_val, labels_val)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, val_dataloader, data


# Define data visualization function
def visualize_data(data):
    '''
    visualizes 16 random images from dataset
    '''
    
    data_length = len(data)
    num_indices = 16
    
    # Generate 15 unique random indices using numpy
    random_indices = np.random.choice(data_length, size=num_indices, replace=False)

    #plot the examples for source
    fig1=plt.figure(figsize=(8,8))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")

        img = data[random_indices[i]]
        example_image = make_lupton_rgb(img[2], img[1], img[0]) #change band by switching 0:1 to 1:2 or 2:3

        plt.imshow(example_image, aspect='auto')



# Define and initialize model
class NeuralNetwork(nn.Module):
    def __init__(self, npix):
        super(NeuralNetwork, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(8))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc1', nn.Linear(in_features=32*5*5, out_features=128))
        self.regressor.add_module('r_relu1', nn.ReLU(True))
        self.regressor.add_module('r_fc3', nn.Linear(in_features=128, out_features=1))

        self.npix = npix

    def forward(self, x):
        x = x.view(-1, 3, self.npix, self.npix)

        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        estimate = self.regressor(features)
        estimate = F.relu(estimate)
        estimate = estimate.view(-1)

        return estimate, features


    def get_feature(self, x):
        x = x.view(-1, 3, self.npix, self.npix)
        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        return features


# Define and initialize model
class NeuralNetworkMVE(nn.Module):
    def __init__(self, npix):
        super(NeuralNetworkMVE, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(8))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc1', nn.Linear(in_features=32*5*5, out_features=128))
        self.regressor.add_module('r_sig1', nn.Sigmoid())
        self.regressor.add_module('r_fc3', nn.Linear(in_features=128, out_features=2))

        self.npix = npix

    def forward(self, x):
        x = x.view(-1, 3, self.npix, self.npix)

        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        estimate = self.regressor(features)
        estimate = F.relu(estimate)
        estimate = estimate.view(-1, 2)

        return estimate, features


    def get_feature(self, x):
        x = x.view(-1, 3, self.npix, self.npix)
        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        return features


# Define and initialize model
class NeuralNetworkMVEv2(nn.Module):
    def __init__(self, npix):
        super(NeuralNetworkMVEv2, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(8))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc1', nn.Linear(in_features=32*5*5, out_features=128))
        self.regressor.add_module('f_relu1', nn.ReLU(True))
        self.regressor.add_module('r_fc2', nn.Linear(in_features=128, out_features=32))
        self.regressor.add_module('r_sig2', nn.Sigmoid())
        self.regressor.add_module('r_fc3', nn.Linear(in_features=32, out_features=2))

        self.npix = npix

    def forward(self, x):
        x = x.view(-1, 3, self.npix, self.npix)

        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        estimate = self.regressor(features)
        estimate = F.relu(estimate)
        estimate = estimate.view(-1, 2)

        return estimate, features


    def get_feature(self, x):
        x = x.view(-1, 3, self.npix, self.npix)
        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        return features

# Define and initialize model
class NeuralNetworkMVEv3(nn.Module):
    def __init__(self, npix):
        super(NeuralNetworkMVEv3, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(8))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc1', nn.Linear(in_features=32*5*5, out_features=128))
        self.regressor.add_module('f_sig1', nn.Sigmoid())
        self.regressor.add_module('r_fc2', nn.Linear(in_features=128, out_features=32))
        self.regressor.add_module('r_sig2', nn.Sigmoid())
        self.regressor.add_module('r_fc3', nn.Linear(in_features=32, out_features=2))

        self.npix = npix

    def forward(self, x):
        x = x.view(-1, 3, self.npix, self.npix)

        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        estimate = self.regressor(features)
        estimate = F.relu(estimate)
        estimate = estimate.view(-1, 2)

        return estimate, features


    def get_feature(self, x):
        x = x.view(-1, 3, self.npix, self.npix)
        features = self.feature(x)
        features = features.view(-1, 32*5*5)
        return features

# Code from https://github.com/ZongxianLee/MMD_Loss.Pytorch
# Comments by Shrihan Agarwal

class MMD_loss(nn.Module):
    """
    Calculate the MMD Loss using a Gaussian Kernel.
    MMD is the distance between the mean embeddings of the source/target dataset. 
    The distances are determined using a Gaussian kernel 
        k(x, y) ~ exp(-(x-y)^2 / (2 * sigma)).

    The bandwidth sigma is either input as fixed in `fix_sigma` or determined dynamically.
    One bandwidth is insufficient - small sigma leads to localization, and large leads to spread.
    Need to capture similarities at various scales.

    E.g. kernel_mul = 2, kernel_num = 5, fix_sigma = 1 creates:
    bandwidth_list = [1, 2, 4, 8, 16]

    Then uses pairwise kernel distances to compute MMD loss:
        Loss = Mean(source pairwise + target pairwise - source/target - target/source)
    
    """
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num # Number of kernels to use
        self.kernel_mul = kernel_mul # How much to multiply the kernel by to get a new kernel
        self.fix_sigma = None
        return
        
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])

        # Concatenate source and target catalogs along batch dimension
        # Source: (n, d), Target: (m, d), Total: (n + m, d)
        total = torch.cat([source, target], dim=0)

        # Replicate and calculate L2 distances between 
        # all samples, independent of source/target
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        # L2 distance is of shape (n + m, n + m)
        L2_distance = ((total0-total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)

        # Create bandwidth list as described
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # Calculate kernel based distance using list of bandwidths and aggregate
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # Return the kernel matrix which is of shape (n + m, n + m), summing over bandwidths
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size] # source pairwise kernel dists
        YY = kernels[batch_size:, batch_size:] # target pairwise kernel dists
        XY = kernels[:batch_size, batch_size:] # between source and target samples
        YX = kernels[batch_size:, :batch_size] # between source and target samples
        loss = torch.mean(XX + YY - XY - YX)    # definition of MMD loss
        return loss

def loss_bnll(mean, variance, truth, beta, epsilon = 1e-7):  # beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param truth: truth of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    variance = variance + epsilon
    loss = 0.5 * ((truth - mean) ** 2 / variance + variance.log())
    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    return loss.sum(axis=-1) / len(mean)

# Define training loop
def train_loop(source_dataloader, 
               target_dataloader, 
               model, 
               regressor_loss_fn,
               da_loss,
               optimizer,
               n_epoch,
               epoch,
               init_wt,
               final_wt):
    """
    Trains the Neural Network on Source/Target Domains with the following loss:
        Loss = Source Regression Loss + 1.4 * DA MMD Loss
    
    source_dataloader: DataLoader for the source domain data.
	target_dataloader: DataLoader for the target domain data.
	model: The neural network model to be trained.
	regressor_loss_fn: Loss function for the regression task (e.g., MSELoss).
	da_loss: Loss function for domain adaptation (e.g., MMD loss).
	optimizer: Optimizer for the model parameters.
	n_epoch: Total number of epochs for training.
	epoch: Current epoch number.
    """

    domain_error = 0
    domain_classifier_accuracy = 0
    estimator_error = 0
    score_list = np.array([])

    # Iteration length is shorter of the two datasets
    len_dataloader = min(len(source_dataloader), len(target_dataloader))
    data_source_iter = iter(source_dataloader)
    data_target_iter = iter(target_dataloader)

    # Iterate over the two datasets
    i = 0
    while i < len_dataloader:

        # Time-varying hyperparameter, p 0 -> infty, alpha 0 -> 1
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader # UNUSED
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 # UNUSED

        # Source Training

        # Load a batch of source data, move to GPU
        data_source = next(data_source_iter)
        X, y = data_source
        X = X.float()
        X = X.cuda()
        y = y.cuda()

        # Zero model gradients and labels
        model.zero_grad()
        batch_size = len(y)

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()
        domain_label = domain_label.cuda()

        # Apply data to model and get predictions, embeddings, apply gradients
        estimate_output, domain_output_source = model(X)

        # Calculate source regression loss based on predictions
        estimate_loss = regressor_loss_fn(estimate_output, y)

        # Target Training

        data_target = next(data_target_iter)
        X_target, _ = data_target
        X_target = X_target.float()
        X_target = X_target.cuda()

        batch_size = len(X_target)

        _, domain_output_target = model(X_target)

        # Calculate the DA Loss between source and target, MMD loss
        domain_loss = da_loss(domain_output_source, domain_output_target)

        # Hyperparameter of 1.4 set to weight domain loss vs source loss
        # Perhaps this is where alpha was initially used
        da_weight = init_wt - ((init_wt - final_wt) * (epoch / n_epoch))
        loss = estimate_loss + domain_loss * da_weight

        # Backpropagation, update optimizer lr
        loss.backward()
        optimizer.step()

        # Update values
        
        # Domain loss is the DA loss or MMD loss between embedding outputs
        domain_error += domain_loss.item()

        # Estimator loss is the source data loss on regression
        estimator_error += estimate_loss.item()

        # Calculate the R2 score of the predictions vs. labels
        score = r2_score(y.cpu().detach().numpy(), estimate_output.cpu().detach().numpy())
        score_list = np.append(score_list, score)

        i += 1

    # Calculate average scores/errors of batches for this epoch
    score = np.mean(score_list)
    domain_error = domain_error / (len_dataloader)
    estimator_error /= len_dataloader

    return [domain_error, estimator_error, score]


# Define training loop
def train_loop_mve(source_dataloader, 
               target_dataloader, 
               model, 
               regressor_loss_fn,
               da_loss,
               optimizer,
               da_weight,
               beta_val):
    """
    Trains the Neural Network on Source/Target Domains with the following loss:
        Loss = Source Regression Loss + 1.4 * DA MMD Loss
    
    source_dataloader: DataLoader for the source domain data.
	target_dataloader: DataLoader for the target domain data.
	model: The neural network model to be trained.
	regressor_loss_fn: Loss function for the regression task (e.g., MSELoss).
	da_loss: Loss function for domain adaptation (e.g., MMD loss).
	optimizer: Optimizer for the model parameters.
    """

    domain_error = 0
    domain_classifier_accuracy = 0
    estimator_error = 0
    mve_error = 0
    score_list = np.array([])

    # Iteration length is shorter of the two datasets
    len_dataloader = min(len(source_dataloader), len(target_dataloader))
    data_source_iter = iter(source_dataloader)
    data_target_iter = iter(target_dataloader)

    # Iterate over the two datasets
    i = 0
    while i < len_dataloader:

        # Source Training

        # Load a batch of source data, move to GPU
        data_source = next(data_source_iter)
        X, y = data_source
        X = X.float()
        X = X.cuda()
        y = y.cuda()

        # Zero model gradients and labels
        model.zero_grad()
        batch_size = len(y)

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()
        domain_label = domain_label.cuda()

        # Apply data to model and get predictions, embeddings, apply gradients
        estimate_output, domain_output_source = model(X)
        mean = estimate_output[:, 0]
        variance = estimate_output[:, 1]

        # Calculate source regression loss based on predictions
        estimate_loss = regressor_loss_fn(mean, y)

        # Target Training

        data_target = next(data_target_iter)
        X_target, _ = data_target
        X_target = X_target.float()
        X_target = X_target.cuda()

        batch_size = len(X_target)

        _, domain_output_target = model(X_target)

        # Calculate the DA Loss between source and target, MMD loss
        with torch.no_grad():
            domain_loss = da_loss(domain_output_source, domain_output_target)
            score = r2_score(y.cpu().detach().numpy(), mean.cpu().detach().numpy())
            
        mve_loss = loss_bnll(mean.flatten(), variance.flatten(), y, beta = beta_val)
        
        # Loss is combination of mve and domain loss, weighted by da_weight
        loss = mve_loss #+ domain_loss * da_weight 

        
        # Backpropagation, update optimizer lr
        loss.backward()
        optimizer.step()

        # Update values
        
        # Domain loss is the DA loss or MMD loss between embedding outputs
        # Estimator loss is the source data loss on regression
        # MVE Error is the mve loss on training 
        estimator_error += estimate_loss.item()
        domain_error += domain_loss.item()
        mve_error += mve_loss.item()
        score_list = np.append(score_list, score)
        
        i += 1

    # Calculate average scores/errors of batches for this epoch
    score = np.mean(score_list)
    domain_error = domain_error / (len_dataloader)
    estimator_error /= len_dataloader
    mve_error /= len_dataloader

    return [domain_error, estimator_error, mve_error, score]



# Define testing loop

def test_loop(source_dataloader, 
              target_dataloader, 
              model, 
              regressor_loss_fn, 
              da_loss, 
              n_epoch, 
              epoch):
    """
    Tests the model accuracy.
    
    source_dataloader: DataLoader for the source domain data.
	target_dataloader: DataLoader for the target domain data.
	model: The neural network model to be trained.
	regressor_loss_fn: Loss function for the regression task (e.g., MSELoss).
	da_loss: Loss function for domain adaptation (e.g., MMD loss). UNUSED
	n_epoch: Total number of epochs for training.
	epoch: Current epoch number.
    """

    
    # Evaluating without gradient computation in bg for validation
    with torch.no_grad():
        
        len_dataloader = min(len(source_dataloader), len(target_dataloader))
        data_source_iter = iter(source_dataloader)
        data_target_iter = iter(target_dataloader)

        
        domain_classifier_error = 0
        domain_classifier_accuracy = 0
        estimator_error = 0
        estimator_error_target = 0
        score_list = np.array([])
        score_list_target = np.array([])

        i = 0
        while i < len_dataloader:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Source Testing

            data_source = next(data_source_iter)
            X, y = data_source
            X = X.float()
            X = X.cuda()
            y = y.cuda()

            batch_size = len(y)

            estimate_output, domain_output = model(X)

            estimate_loss = regressor_loss_fn(estimate_output, y)

            # Target Testing

            data_target = next(data_target_iter)
            X_target, y_target = data_target
            X_target = X_target.float()
            X_target = X_target.cuda()
            y_target = y_target.cuda()

            batch_size = len(X_target)

            estimate_output_target, domain_output = model(X_target)

            estimate_loss_target = regressor_loss_fn(estimate_output_target, y_target)

            # Update values

            # Regression loss on validation testing
            estimator_error += estimate_loss.item()
            estimator_error_target += estimate_loss_target.item()

            # R2 Scores on validation testing
            score = r2_score(y.cpu(), estimate_output.cpu())
            score_list = np.append(score_list, score)
            score_target = r2_score(y_target.cpu(), estimate_output_target.cpu())
            score_list_target = np.append(score_list_target, score_target)

            i += 1

        score = np.mean(score_list)
        score_target = np.mean(score_list_target)
        estimator_error /= len_dataloader
        estimator_error_target /= len_dataloader
        
    classifier_error = 1
    return [classifier_error, estimator_error, estimator_error_target, score, score_target]

def test_loop_mve(source_dataloader, 
              target_dataloader, 
              model, 
              regressor_loss_fn, 
              beta_val):
    """
    Tests the model accuracy.
    
    source_dataloader: DataLoader for the source domain data.
	target_dataloader: DataLoader for the target domain data.
	model: The neural network model to be trained.
	regressor_loss_fn: Loss function for the regression task (e.g., MSELoss).
    """

    
    # Evaluating without gradient computation in bg for validation
    with torch.no_grad():
        
        len_dataloader = min(len(source_dataloader), len(target_dataloader))
        data_source_iter = iter(source_dataloader)
        data_target_iter = iter(target_dataloader)

        
        domain_classifier_error = 0
        domain_classifier_accuracy = 0
        estimator_error = 0
        estimator_error_target = 0
        mve_error = 0
        mve_error_target = 0
        nll_error = 0
        nll_error_target = 0
        score_list = np.array([])
        score_list_target = np.array([])

        i = 0
        while i < len_dataloader:

            # Source Testing

            data_source = next(data_source_iter)
            X, y = data_source
            X = X.float()
            X = X.cuda()
            y = y.cuda()

            batch_size = len(y)

            estimate_output, domain_output = model(X)
            source_mean = estimate_output[:, 0]
            source_variance = estimate_output[:, 1]
            
            estimate_loss = regressor_loss_fn(source_mean, y)
            mve_loss = loss_bnll(source_mean.flatten(), source_variance.flatten(), y, beta = beta_val)
            nll_loss = loss_bnll(source_mean.flatten(), source_variance.flatten(), y, beta = 0.0)

            # Target Testing

            data_target = next(data_target_iter)
            X_target, y_target = data_target
            X_target = X_target.float()
            X_target = X_target.cuda()
            y_target = y_target.cuda()

            batch_size = len(X_target)

            estimate_output_target, domain_output = model(X_target)
            target_mean = estimate_output_target[:, 0]
            target_variance = estimate_output_target[:, 1]
            
            estimate_loss_target = regressor_loss_fn(target_mean, y_target)
            mve_loss_target = loss_bnll(target_mean.flatten(), target_variance.flatten(), y_target, beta = beta_val)
            nll_loss_target = loss_bnll(target_mean.flatten(), target_variance.flatten(), y_target, beta = 0.0)
            
            # Update values

            # Regression loss on validation testing
            estimator_error += estimate_loss.item()
            estimator_error_target += estimate_loss_target.item()

            # MVE loss on validation testing
            mve_error += mve_loss.item()
            mve_error_target += mve_loss_target.item()

            # NLL loss on validation testing
            nll_error += nll_loss.item()
            nll_error_target += nll_loss_target.item()

            # R2 Scores on validation testing
            score = r2_score(y.cpu(), source_mean.cpu())
            score_list = np.append(score_list, score)
            score_target = r2_score(y_target.cpu(), target_mean.cpu())
            score_list_target = np.append(score_list_target, score_target)

            i += 1

        score = np.mean(score_list)
        score_target = np.mean(score_list_target)
        estimator_error /= len_dataloader
        estimator_error_target /= len_dataloader
        mve_error /= len_dataloader
        mve_error_target /= len_dataloader
        nll_error /= len_dataloader
        nll_error_target /= len_dataloader
        
    return [estimator_error, estimator_error_target, score, score_target, mve_error, mve_error_target, nll_error, nll_error_target]

def initialize_state(mod_name, model, optimizer):
    stats = {'train_DA_loss':[],
                 'train_regression_loss':[],
                 'train_mve_loss':[],
                 'train_r2_score':[],
                 'val_source_regression_loss':[],
                 'val_target_regression_loss':[],
                 'val_source_r2_score':[],
                 'val_target_r2_score':[],
                 'val_source_mve_loss': [],
                 'val_target_mve_loss': [],
                 'val_source_nll_loss': [],
                 'val_target_nll_loss': [],
                 'da_weight': [],
                 'beta': [],
                'epoch_no': 0}

    best_target_R2 = -1.0
    best_mve_loss = 1e6
    best_nll_loss = 1e6
    best_snll_loss = 1e6
    
    if mod_name is not None:
        state = torch.load(mod_name)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        
        stat_file = Path(mod_name+'.json')
        if stat_file.is_file():
            stats = json.load(open(mod_name+'.json', 'r'))

        best_target_R2 = max(stats['val_target_r2_score'])
        best_mve_loss = min(stats['val_target_mve_loss'])
        best_nll_loss = min(stats['val_target_nll_loss'])
        best_snll_loss = min(stats['val_source_nll_loss'])
    
    return stats, model, optimizer, best_target_R2, best_mve_loss, best_nll_loss, best_snll_loss


def save_model(mod_name, model, optimizer, stats):
    state = {
            'epoch': stats['epoch_no'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    json.dump(stats, open(mod_name+'.json', 'w'))
    torch.save(state, mod_name)
    return True

def print_epoch_scores(stats, epoch, t):
    """ Prints all relevant scores for each epoch. """
    train_stats = [i for i in stats.keys() if "train" in i]
    val_stats = [i for i in stats.keys() if "val" in i]
    fmt = lambda k: " ".join([i.capitalize() for i in k.split('_')]) + ": "
    
    print("\nEpoch {0}: {1:.2f}s".format(epoch, t) + "\n-------------------------------")
    print(" Training Statistics:")
    for s in train_stats:
        print("\t" + fmt(s) + ": {:.4f}".format(stats[s][-1]))
    print(" Validation Statistics:")
    for s in val_stats:
        print("\t" + fmt(s) + ": {:.4f}".format(stats[s][-1]))


def generate_isomaps(source_data, target_data, model, n_neighbors = 5, n_components = 2, n_points = 1000):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    
    train_isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    
    with torch.no_grad():
        source_tensor = torch.FloatTensor(source_data[:n_points]).to(device)
        target_tensor = torch.FloatTensor(target_data[:n_points]).to(device)
        sfeat = model.get_feature(source_tensor).cpu().detach().numpy()
        tfeat = model.get_feature(target_tensor).cpu().detach().numpy()
        catfeat = np.concatenate((sfeat, tfeat), axis=0)
        
    train_isomap = train_isomap.fit(catfeat)
    trained_source_iso = train_isomap.transform(sfeat)
    trained_target_iso = train_isomap.transform(tfeat)
    
    del sfeat
    del tfeat

    return trained_source_iso, trained_target_iso

def show_isomaps(source_iso, 
                 target_iso, 
                 source_labels,
                 target_labels,
                 mod_name, 
                 name = "viz",
                 axlim = 50,
                 save = False):
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    ax1, ax2, ax3 = axes

    # Superimpose the source and target isomap to check similarity
    
    ax1.scatter(source_iso[:, 0], source_iso[:, 1], s=3, marker='o', alpha = 0.5)
    ax1.scatter(target_iso[:, 0], target_iso[:, 1], s=3, marker='^', alpha = 0.5)
    ax1.set_xlim(-axlim, axlim)
    ax1.set_ylim(-axlim, axlim)
    ax1.set_title('Source and Target')

    scatter2 = ax2.scatter(source_iso[:, 0], source_iso[:, 1], s=3, c = source_labels)
    ax2.set_xlim(-axlim, axlim)
    ax2.set_ylim(-axlim, axlim)
    ax2.set_title('Source Embedding')

    scatter3 = ax3.scatter(target_iso[:, 0], target_iso[:, 1], s=3, c = target_labels)
    ax3.set_xlim(-axlim, axlim)
    ax3.set_ylim(-axlim, axlim)
    ax3.set_title('Target Embedding')

    cbar = fig.colorbar(scatter2, ax=[ax1, ax2, ax3], orientation='vertical')
    cbar.set_label('$\\theta_E$')

    for i in axes.ravel():
        i.set_xlabel('Component 1')
        i.set_ylabel('Component 2')
    
    if save:
        plt.savefig(mod_name + "_" + str(name) + "_isomap.png", bbox_inches = 'tight', dpi = 400)

    plt.show()
    
    return fig, axes