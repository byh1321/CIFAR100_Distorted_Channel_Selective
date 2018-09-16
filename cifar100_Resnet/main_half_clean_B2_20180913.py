"""
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
"""

from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse
#import VGG16
#import Resnet_vision as RS
import Resnet34 as RS2
import Resnet18 as RS

import cifar_dirty_test
import cifar_dirty_train
import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--network', default='NULL', help='input network ckpt name', metavar="FILE")
parser.add_argument('--outputfile', default='garbage.txt', help='output file name', metavar="FILE")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
									  transforms.RandomHorizontalFlip(),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
cifar_train = dset.CIFAR100("./", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("./", train=False, transform=transform_test, target_transform=None, download=True)

cifar_test_gaussian_025 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_0.0_test_targets.csv")
cifar_test_gaussian_016 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.16_blur_0.0_test_targets.csv")
cifar_test_gaussian_008 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.08_blur_0.0_test_targets.csv")

cifar_train_gaussian_025 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_0.0_train_targets.csv")
cifar_train_gaussian_016 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.16_blur_0.0_train_targets.csv")
cifar_train_gaussian_008 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.08_blur_0.0_train_targets.csv")

cifar_test_blur_10 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_1.0_test_targets.csv")
cifar_test_blur_09 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.9_test_targets.csv")
cifar_test_blur_08 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.0_blur_0.8_test_targets.csv")
cifar_test_blur_0675 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.675_test_targets.csv")
cifar_test_blur_06 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.6_test_targets.csv")
cifar_test_blur_05 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.5_test_targets.csv")
cifar_test_blur_045 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.45_test_targets.csv")
cifar_test_blur_04 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.4_test_targets.csv")
cifar_test_blur_03 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.3_test_targets.csv")
cifar_test_blur_066 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.66_test_targets.csv")
cifar_test_blur_033 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.33_test_targets.csv")

cifar_train_blur_10 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_1.0_train_targets.csv")
cifar_train_blur_09 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.9_train_targets.csv")
cifar_train_blur_08 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.0_blur_0.8_train_targets.csv")
cifar_train_blur_0675 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.675_train_targets.csv")
cifar_train_blur_06 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.6_train_targets.csv")
cifar_train_blur_05 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.5_train_targets.csv")
cifar_train_blur_045 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.45_train_targets.csv")
cifar_train_blur_04 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.4_train_targets.csv")
cifar_train_blur_03 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.3_train_targets.csv")
cifar_train_blur_066 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.66_train_targets.csv")
cifar_train_blur_033 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.33_train_targets.csv")

cifar_train_gaussian_025 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_0.0_train_targets.csv")
cifar_train_blur_10 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_1.0_train_targets.csv")

cifar_train_gaussian_008_blur_03_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.08_blur_0.3_train_targets.csv") 
cifar_train_gaussian_016_blur_06_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.16_blur_0.6_train_targets.csv") 
cifar_train_gaussian_008_blur_033_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.08_blur_0.33_train_targets.csv") 
cifar_train_gaussian_016_blur_066_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.16_blur_0.66_train_targets.csv") 
cifar_train_gaussian_016_blur_08_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.16_blur_0.8_train_targets.csv") 
cifar_train_gaussian_025_blur_10_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_1.0_train_targets.csv") 
#train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([cifar_train, cifar_train_blur_0675]),batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test_blur_0675,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
		)
		self.layer1_basic1 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer1_basic3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer2_basic1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_downsample = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_basic2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer2_basic3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_basic4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic1 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_downsample = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_basic2 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic3 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_basic4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer4_basic1 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_downsample = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer4_basic2 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer4_basic3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_basic4 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.linear = nn.Sequential(
			nn.Linear(512, 100, bias=False)
		)
		self._initialize_weights()

	def forward(self,x):
		if args.fixed:
			x = quant(x)
			x = roundmax(x)

		out = x.clone()
		out = self.conv1(out)

		residual = out

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic1(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic2(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu1(out)
		residual = out

		out = self.layer1_basic3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic4(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu2(out)
		residual = self.layer2_downsample(out)

		out = self.layer2_basic1(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic2(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual

		out = self.layer2_relu1(out)
		residual = out

		out = self.layer2_basic3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic4(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu2(out)

		residual = self.layer3_downsample(out)

		out = self.layer3_basic1(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic2(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu1(out)

		residual = out

		out = self.layer3_basic3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic4(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu2(out)

		residual = self.layer4_downsample(out)

		out = self.layer4_basic1(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic2(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)
		out += residual
		out = self.layer4_relu1(out)
		residual = out

		out = self.layer4_basic3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic4(out)

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu2(out)
		out = F.avg_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		#print(out.size())
		out = self.linear(out)

		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				#print(m)
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

				#if m.bias is not None:
					#nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				#nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				#print(m)
				nn.init.normal_(m.weight, 0, 0.01)
				#nn.init.constant_(m.bias, 0)

def roundmax(input):
	maximum = 2**args.iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	return input	

def quant(input):
	input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return input

def set_mask(mask, block, val):
	if block == 0:
		mask[0][:,:,:,:] = val
		mask[1][:,:,:,:] = val 
		mask[2][:,:,:,:] = val 
		mask[3][:,:,:,:] = val
		mask[4][:,:,:,:] = val
		mask[5][:,:,:,:] = val
		mask[6][:,:,:,:] = val
		mask[7][:,:,:,:] = val
		mask[8][:,:,:,:] = val
		mask[9][:,:,:,:] = val
		mask[10][:,:,:,:] = val
		mask[11][:,:,:,:] = val
		mask[12][:,:,:,:] = val
		mask[13][:,:,:,:] = val
		mask[14][:,:,:,:] = val
		mask[15][:,:,:,:] = val
		mask[16][:,:,:,:] = val
		mask[17][:,:,:,:] = val
		mask[18][:,:,:,:] = val
		mask[19][:,:,:,:] = val
		mask[20][:,:] = val 
	elif block == 1:
		mask[0][0:55,:,:,:] = val
		mask[1][0:55,0:55,:,:] = val 
		mask[2][0:55,0:55,:,:] = val 
		mask[3][0:55,0:55,:,:] = val
		mask[4][0:55,0:55,:,:] = val

		mask[5][0:111,0:55,:,:] = val
		mask[6][0:111,0:111,:,:] = val
		mask[7][0:111,0:111,:,:] = val
		mask[8][0:111,0:111,:,:] = val

		mask[9][0:223,0:111,:,:] = val
		mask[10][0:223,0:223,:,:] = val
		mask[11][0:223,0:223,:,:] = val
		mask[12][0:223,0:223,:,:] = val

		mask[13][0:447,0:223,:,:] = val
		mask[14][0:447,0:447,:,:] = val
		mask[15][0:447,0:447,:,:] = val
		mask[16][0:447,0:447,:,:] = val

		mask[17][0:111,0:55,:,:] = val
		mask[18][0:223,0:111,:,:] = val
		mask[19][0:447,0:223,:,:] = val

		mask[20][:,0:447] = val 
	elif block == 2:
		mask[0][0:47,:,:,:] = val
		mask[1][0:47,0:47,:,:] = val 
		mask[2][0:47,0:47,:,:] = val 
		mask[3][0:47,0:47,:,:] = val
		mask[4][0:47,0:47,:,:] = val

		mask[5][0:95,0:47,:,:] = val
		mask[6][0:95,0:95,:,:] = val
		mask[7][0:95,0:95,:,:] = val
		mask[8][0:95,0:95,:,:] = val

		mask[9][0:191,0:95,:,:] = val
		mask[10][0:191,0:191,:,:] = val
		mask[11][0:191,0:191,:,:] = val
		mask[12][0:191,0:191,:,:] = val

		mask[13][0:383,0:191,:,:] = val
		mask[14][0:383,0:383,:,:] = val
		mask[15][0:383,0:383,:,:] = val
		mask[16][0:383,0:383,:,:] = val

		mask[17][0:95,0:47,:,:] = val
		mask[18][0:191,0:95,:,:] = val
		mask[19][0:383,0:191,:,:] = val

		mask[20][:,0:383] = val 
	elif block == 3:
		mask[0][0:39,:,:,:] = val
		mask[1][0:39,0:39,:,:] = val 
		mask[2][0:39,0:39,:,:] = val 
		mask[3][0:39,0:39,:,:] = val
		mask[4][0:39,0:39,:,:] = val

		mask[5][0:79,0:39,:,:] = val
		mask[6][0:79,0:79,:,:] = val
		mask[7][0:79,0:79,:,:] = val
		mask[8][0:79,0:79,:,:] = val

		mask[9][0:159,0:79,:,:] = val
		mask[10][0:159,0:159,:,:] = val
		mask[11][0:159,0:159,:,:] = val
		mask[12][0:159,0:159,:,:] = val

		mask[13][0:319,0:159,:,:] = val
		mask[14][0:319,0:319,:,:] = val
		mask[15][0:319,0:319,:,:] = val
		mask[16][0:319,0:319,:,:] = val

		mask[17][0:79,0:39,:,:] = val
		mask[18][0:159,0:79,:,:] = val
		mask[19][0:319,0:159,:,:] = val

		mask[20][:,0:319] = val 
	elif block == 4:
		mask[0][0:31,:,:,:] = val
		mask[1][0:31,0:31,:,:] = val 
		mask[2][0:31,0:31,:,:] = val 
		mask[3][0:31,0:31,:,:] = val
		mask[4][0:31,0:31,:,:] = val

		mask[5][0:63,0:31,:,:] = val
		mask[6][0:63,0:63,:,:] = val
		mask[7][0:63,0:63,:,:] = val
		mask[8][0:63,0:63,:,:] = val

		mask[9][0:127,0:63,:,:] = val
		mask[10][0:127,0:127,:,:] = val
		mask[11][0:127,0:127,:,:] = val
		mask[12][0:127,0:127,:,:] = val

		mask[13][0:255,0:127,:,:] = val
		mask[14][0:255,0:255,:,:] = val
		mask[15][0:255,0:255,:,:] = val
		mask[16][0:255,0:255,:,:] = val

		mask[17][0:63,0:31,:,:] = val
		mask[18][0:127,0:63,:,:] = val
		mask[19][0:255,0:127,:,:] = val

		mask[20][:,0:255] = val 
	return mask

def save_network(layer):
	for child in net2.children():
		for param in child.conv1[0].parameters():
			layer[0] = param.data
	for child in net2.children():
		for param in child.layer1_basic1[0].parameters():
			layer[1] = param.data		
	for child in net2.children():
		for param in child.layer1_basic2[0].parameters():
			layer[2] = param.data		
	for child in net2.children():
		for param in child.layer1_basic3[0].parameters():
			layer[3] = param.data		
	for child in net2.children():
		for param in child.layer1_basic4[0].parameters():
			layer[4] = param.data	
	for child in net2.children():
		for param in child.layer2_basic1[0].parameters():
			layer[5] = param.data
	for child in net2.children():
		for param in child.layer2_basic2[0].parameters():
			layer[6] = param.data
	for child in net2.children():
		for param in child.layer2_basic3[0].parameters():
			layer[7] = param.data
	for child in net2.children():
		for param in child.layer2_basic4[0].parameters():
			layer[8] = param.data
	for child in net2.children():
		for param in child.layer3_basic1[0].parameters():
			layer[9] = param.data
	for child in net2.children():
		for param in child.layer3_basic2[0].parameters():
			layer[10] = param.data
	for child in net2.children():
		for param in child.layer3_basic3[0].parameters():
			layer[11] = param.data
	for child in net2.children():
		for param in child.layer3_basic4[0].parameters():
			layer[12] = param.data
	for child in net2.children():
		for param in child.layer4_basic1[0].parameters():
			layer[13] = param.data
	for child in net2.children():
		for param in child.layer4_basic2[0].parameters():
			layer[14] = param.data
	for child in net2.children():
		for param in child.layer4_basic3[0].parameters():
			layer[15] = param.data
	for child in net2.children():
		for param in child.layer4_basic4[0].parameters():
			layer[16] = param.data
	for child in net2.children():
		for param in child.layer2_downsample[0].parameters():
			layer[17] = param.data
	for child in net2.children():
		for param in child.layer3_downsample[0].parameters():
			layer[18] = param.data
	for child in net2.children():
		for param in child.layer4_downsample[0].parameters():
			layer[19] = param.data
	for child in net2.children():
		for param in child.linear[0].parameters():
			layer[20] = param.data
	return layer

def add_network():
	layer = torch.load('mask_null.dat')
	layer = save_network(layer)
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,layer[0])
	for child in net.children():
		for param in child.layer1_basic1[0].parameters():
			param.data = torch.add(param.data,layer[1])		
	for child in net.children():
		for param in child.layer1_basic2[0].parameters():
			param.data = torch.add(param.data,layer[2])		
	for child in net.children():
		for param in child.layer1_basic3[0].parameters():
			param.data = torch.add(param.data,layer[3])		
	for child in net.children():
		for param in child.layer1_basic4[0].parameters():
			param.data = torch.add(param.data,layer[4])	
	for child in net.children():
		for param in child.layer2_basic1[0].parameters():
			param.data = torch.add(param.data,layer[5])
	for child in net.children():
		for param in child.layer2_basic2[0].parameters():
			param.data = torch.add(param.data,layer[6])
	for child in net.children():
		for param in child.layer2_basic3[0].parameters():
			param.data = torch.add(param.data,layer[7])
	for child in net.children():
		for param in child.layer2_basic4[0].parameters():
			param.data = torch.add(param.data,layer[8])
	for child in net.children():
		for param in child.layer3_basic1[0].parameters():
			param.data = torch.add(param.data,layer[9])
	for child in net.children():
		for param in child.layer3_basic2[0].parameters():
			param.data = torch.add(param.data,layer[10])
	for child in net.children():
		for param in child.layer3_basic3[0].parameters():
			param.data = torch.add(param.data,layer[11])
	for child in net.children():
		for param in child.layer3_basic4[0].parameters():
			param.data = torch.add(param.data,layer[12])
	for child in net.children():
		for param in child.layer4_basic1[0].parameters():
			param.data = torch.add(param.data,layer[13])
	for child in net.children():
		for param in child.layer4_basic2[0].parameters():
			param.data = torch.add(param.data,layer[14])
	for child in net.children():
		for param in child.layer4_basic3[0].parameters():
			param.data = torch.add(param.data,layer[15])
	for child in net.children():
		for param in child.layer4_basic4[0].parameters():
			param.data = torch.add(param.data,layer[16])
	for child in net.children():
		for param in child.layer2_downsample[0].parameters():
			param.data = torch.add(param.data,layer[17])
	for child in net.children():
		for param in child.layer3_downsample[0].parameters():
			param.data = torch.add(param.data,layer[18])
	for child in net.children():
		for param in child.layer4_downsample[0].parameters():
			param.data = torch.add(param.data,layer[19])
	for child in net.children():
		for param in child.linear[0].parameters():
			param.data = torch.add(param.data,layer[20])
	return layer

def net_mask_mul(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.layer1_basic1[0].parameters():
			param.data = torch.mul(param.data,mask[1].cuda())		
	for child in net.children():
		for param in child.layer1_basic2[0].parameters():
			param.data = torch.mul(param.data,mask[2].cuda())		
	for child in net.children():
		for param in child.layer1_basic3[0].parameters():
			param.data = torch.mul(param.data,mask[3].cuda())		
	for child in net.children():
		for param in child.layer1_basic4[0].parameters():
			param.data = torch.mul(param.data,mask[4].cuda())	
	for child in net.children():
		for param in child.layer2_basic1[0].parameters():
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.layer2_basic2[0].parameters():
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.layer2_basic3[0].parameters():
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.layer2_basic4[0].parameters():
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.layer3_basic1[0].parameters():
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.layer3_basic2[0].parameters():
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.layer3_basic3[0].parameters():
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.layer3_basic4[0].parameters():
			param.data = torch.mul(param.data,mask[12].cuda())
	for child in net.children():
		for param in child.layer4_basic1[0].parameters():
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.layer4_basic2[0].parameters():
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.layer4_basic3[0].parameters():
			param.data = torch.mul(param.data,mask[15].cuda())
	for child in net.children():
		for param in child.layer4_basic4[0].parameters():
			param.data = torch.mul(param.data,mask[16].cuda())
	for child in net.children():
		for param in child.layer2_downsample[0].parameters():
			param.data = torch.mul(param.data,mask[17].cuda())
	for child in net.children():
		for param in child.layer3_downsample[0].parameters():
			param.data = torch.mul(param.data,mask[18].cuda())
	for child in net.children():
		for param in child.layer4_downsample[0].parameters():
			param.data = torch.mul(param.data,mask[19].cuda())
	for child in net.children():
		for param in child.linear[0].parameters():
			param.data = torch.mul(param.data,mask[20].cuda())

# Model
if args.mode == 0:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180913_half_clean_B2.t0')
	net = checkpoint['net']

elif args.mode == 1:
	checkpoint = torch.load('./checkpoint/ckpt_20180913_half_clean_B2.t0')
	ckpt = torch.load('./checkpoint/ckpt_20180913_half_clean_B1.t0')
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(0,8))
	if args.mode > 0:
		net2.cuda()
		net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

'''
for child in net.children():
	for param in child.conv1[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer1_basic1[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer1_basic2[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer1_basic3[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer1_basic4[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer2_basic1[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer2_basic2[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer2_basic3[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer2_basic4[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_basic1[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_basic2[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_basic3[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_basic4[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer4_basic1[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer4_basic2[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer4_basic3[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer4_basic4[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer2_downsample[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_downsample[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.layer3_downsample[0].parameters():
		print(param.size())
for child in net.children():
	for param in child.linear[0].parameters():
		print(param.size())
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(set_mask(mask_channel, 2, 1), 3, 0)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		net_mask_mul(mask_channel)
		add_network()
		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(mask_channel, 2, 1)
	net_mask_mul(mask_channel)
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		test_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()
		progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		if args.mode == 0:
			pass
		else:
			print('Saving..')
			state = {
				'net': net.module if use_cuda else net,
				'acc': acc,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt_20180913_half_clean_B2.t0')
			best_acc = acc
	
	return acc

# Train+inference vs. Inference
mode = args.mode
if mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)
		test()
elif mode == 0: # only inference
	test()
else:
	pass

