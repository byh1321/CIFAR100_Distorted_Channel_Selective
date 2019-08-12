"""
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
"""

from __future__ import print_function

import numpy as np

import torch
import utils
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

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--ldpr', default=0, type=int, help='pruning') # mode=1 load pruned trained data. mode=0 is trained, but not pruned data
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--prindex', default=0.0005, type=float)
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
									  transforms.RandomHorizontalFlip(),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

cifar_train = dset.CIFAR100("~/Dataset/CIFAR100/", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("~/Dataset/CIFAR100/", train=False, transform=transform_test, target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=1250, shuffle=False,num_workers=8,drop_last=False)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1, bias=False),  
			nn.BatchNorm2d(64),  
			nn.ReLU(inplace=True),
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, padding=1, bias=False),  
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.maxpool1 = nn.Sequential(
			nn.MaxPool2d(2, 2),  
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1, bias=False),  
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, 3, padding=1, bias=False),  
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.maxpool2 = nn.Sequential(
			nn.MaxPool2d(2, 2),  
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, padding=1, bias=False),  
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=1, bias=False),  
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=1, bias=False),  
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.maxpool3 = nn.Sequential(
			nn.MaxPool2d(2, 2),  
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool4 = nn.Sequential(
			nn.MaxPool2d(2, 2),  
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool5 = nn.Sequential(
			nn.MaxPool2d(2, 2)  
		)
		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512, 512, bias=False),  
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512, 512, bias=False),  
			nn.ReLU(inplace=True),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(512, 100, bias=False)  
		)
		self._initialize_weights()

	def compfeature(self, input):
		prout1 = input.view(1, -1)
		filter = torch.ge(prout1, 0.1)
		mfilter = filter.view(-1, 1)
		print(mfilter)
		return mfilter

	def forward(self, x):

		out1 = self.conv1(x)  

		out2 = self.conv2(out1)  
		out3 = self.maxpool1(out2)

		out4 = self.conv3(out3)  

		out5 = self.conv4(out4)  
		out6 = self.maxpool2(out5)

		out7 = self.conv5(out6)  

		out8 = self.conv6(out7)  

		out9 = self.conv7(out8)  
		out10 = self.maxpool3(out9)

		out11 = self.conv8(out10)  

		out12 = self.conv9(out11)  

		out13 = self.conv10(out12)  
		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14)  

		out16 = self.conv12(out15)  

		out17 = self.conv13(out16)  
		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0), -1)

		out20 = self.fc1(out19)  

		out21 = self.fc2(out20)  

		out22 = self.fc3(out21)  

		return out22
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)

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

def quantize():
        for child in net.children():
                for param in child.conv1[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv2[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv3[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv4[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv5[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv6[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv7[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv8[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv9[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv10[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv11[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv12[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.conv13[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

        for child in net.children():
                for param in child.fc1[1].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.fc2[1].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
        for child in net.children():
                for param in child.fc3[0].parameters():
                        param.data = torch.round(param.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20190802.t0')
	best_acc = 0 
	net = checkpoint['net']

else:
	print('==> Building model..')
	net = CNN()

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

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += float(predicted.eq(targets.data).cpu().sum())

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

		test_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += float(predicted.eq(targets.data).cpu().sum())

		progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		state = {
			'net': net.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		if args.mode > 0:
			print('Saving..')
			torch.save(state, './checkpoint/ckpt_20190802.t0')
		best_acc = acc
	
	return acc

# Train+inference vs. Inference
mode = args.mode
if mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)
		if (epoch == 5) and (args.resume == 0):
			utils.maskGen(num_class = 100)
			utils.saveInitialParameter(net)
		test()
elif mode == 0: # only inference
	test()
else:
	pass
