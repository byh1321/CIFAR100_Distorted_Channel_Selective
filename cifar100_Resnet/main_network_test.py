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
parser.add_argument('--pprec', type=int, default=15, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=15, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--modelsel', type=int, default=0, metavar='N',help='choose model')
parser.add_argument('--testsel', type=int, default=0, metavar='N',help='choose testset')
parser.add_argument('--amp', type=int, default=0, metavar='N',help='amp = 1, multiply mask to the feature')
parser.add_argument('--network', default='NULL', help='input network ckpt name', metavar="FILE")

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

cifar_test_gaussian_015 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.15_blur_0.0_test_targets.csv")
cifar_test_gaussian_010 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.1_blur_0.0_test_targets.csv")
cifar_test_gaussian_005 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.05_blur_0.0_test_targets.csv")

cifar_train_gaussian_015 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.15_blur_0.0_train_targets.csv")
cifar_train_gaussian_010 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.1_blur_0.0_train_targets.csv")
cifar_train_gaussian_005 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.05_blur_0.0_train_targets.csv")

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

#train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([cifar_train, cifar_train_gaussian_025, cifar_train_blur_10, cifar_train_gaussian_blur_mixed]),batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
#test_loader = torch.utils.data.DataLoader(cifar_test_blur_10,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
#test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
#test_loader = torch.utils.data.DataLoader(cifar_test_gaussian_025,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

if args.testsel == 0:
	test_loader = torch.utils.data.DataLoader(cifar_test_blur_09,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 1:
	test_loader = torch.utils.data.DataLoader(cifar_test_blur_0675,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 2:
	test_loader = torch.utils.data.DataLoader(cifar_test_blur_045,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 3:
	test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 4:
	test_loader = torch.utils.data.DataLoader(cifar_test_gaussian_005,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 5:
	test_loader = torch.utils.data.DataLoader(cifar_test_gaussian_010,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 6:
	test_loader = torch.utils.data.DataLoader(cifar_test_gaussian_015,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

#mask_amp = torch.ones(1250,512)
#mask_amp[:,256:512] = 1.5

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


# Load checkpoint.
if args.modelsel == 1:
	checkpoint = torch.load('./checkpoint/ckpt_20180425.t0')
elif args.modelsel == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_half_blur.t0')
elif args.modelsel == 3:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.375_blur.t0')
elif args.modelsel == 4:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.25_blur.t0')
elif args.modelsel == 5:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.125_blur.t0')
elif args.modelsel == 6:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.125_gaussian.t0')
elif args.modelsel == 7:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.25_gaussian.t0')
elif args.modelsel == 8:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_0.375_gaussian.t0')
elif args.modelsel == 9:
	checkpoint = torch.load('./checkpoint/ckpt_20180609_half_clean_half_gaussian.t0')
else:
	checkpoint = torch.load('./checkpoint/'+args.network)

best_acc = 0 
net = checkpoint['net']

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(0,8))
	cudnn.benchmark = True

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
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
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
	#	print('Saving..')
		state = {
			'net': net.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		#torch.save(state, './checkpoint/ckpt_20180425.t0')
		best_acc = acc
	
	return acc

# Truncate weight param
pprec = args.pprec
if args.fixed:
        for child in net.children():
                for param in child.conv1[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv2[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv3[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv4[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv5[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv6[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv7[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv8[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv9[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv10[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv11[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv12[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.conv13[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))

        for child in net.children():
                for param in child.fc1[1].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.fc2[1].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
        for child in net.children():
                for param in child.fc3[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))

test()
if args.fixed:
	print("networkname : ",args.network,"pprec, iwidth, arpec : ",args.pprec, args.iwidth, args.aprec)
# Train+inference vs. Inference
#test()

