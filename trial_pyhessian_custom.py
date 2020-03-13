#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from density_plot import get_esd_plot, density_generate
from spectral_cdf import *
from models.resnet import resnet
from pyhessian import hessian

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader = getData(name='cifar10_without_dataaugmentation',
                                    train_bs=args.mini_hessian_batch_size,
                                    test_bs=1)
##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

# get model
model = resnet(num_classes=10,
               depth=20,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)
if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()  # label loss

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")
model.load_state_dict(torch.load(args.resume))

######################################################
# Begin the computation
######################################################

# turn model to eval mode
model.eval()
if batch_num == 1:
    hessian_comp = hessian(model,
                           criterion,
                           data=hessian_dataloader,
                           cuda=args.cuda,
                           record_data=True)
else:
    hessian_comp = hessian(model,
                           criterion,
                           dataloader=hessian_dataloader,
                           cuda=args.cuda,
                           record_data=True)

print(
    '********** finish data loading and begin Hessian computation **********')

# Custom functions
#hessian_comp.test_function()
# h = hessian_comp.sketch(100)
# print(h)
# print(np.trace(h))
# np.save("temp_100sketch", h)

# h = hessian_comp.sketch(50)
# print(np.trace(h))
# np.save("normalized_50sketch", h)
# scdf(h, plotname="SketchNormalized_Normal")
# scdf(h, plotname="SketchNormalized_Log", logx=True)

h = np.load("temp_100sketch.npy")
# print(h)
print("ok")
print(np.linalg.eigvalsh(h))
eigs = np.linalg.eigvalsh(h)
eigs.flip()
frompower = [167.76622009277344, 106.86729431152344, 89.84439086914062, 51.09503936767578, 44.81565856933594, 43.12944412231445, 38.06986618041992, 30.898088455200195, 25.267065048217773, 24.326675415039062, 19.926258087158203, 16.243072509765625, 15.194313049316406, 15.515100479125977, 14.034353256225586, 12.776123046875, 10.012635231018066, 9.320322036743164, 9.862530708312988, 8.594795227050781]
for i in range(20):
  print("({}, {})".format(frompower[i], (eigs[i] - frompower[i])/frompower[i])) 

# scdf(h)
# scdf(h, plotname="Sketch_Normal")
# scdf(h, plotname="Sketch_Log", logx=True)

# density_eigen, density_weight = hessian_comp.density(n_v=3, debug=True)
# get_esd_plot(density_eigen, density_weight)

# density_eigen, density_weight = hessian_comp.density(debug=True)
# print("----------")
# print(density_eigen)
# print(density_weight)
# eigen_density, eigen_values = density_generate(density_eigen, density_weight)
# print("----------")
# print(eigen_density)
# print(sum(eigen_density))
# print(eigen_values)
# density_to_scdf(eigen_density, eigen_values, plotname="Density_Normal")
# density_to_scdf(eigen_density, eigen_values, plotname="Density_Log", logx=True)

# es = hessian_comp.eigenvalues_lanczos(100)
# print(es)
