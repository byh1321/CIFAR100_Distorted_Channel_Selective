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
import Resnet_vision as RS
#import Resnet34 as RS

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference

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

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)


# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180725.t0')
	best_acc = 0 
	net = checkpoint['net']

else:
	print('==> Building model..')
	net = RS.resnet34()

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(0,1))
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
		correct += predicted.eq(targets.data).cpu().sum()

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
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt_20180725.t0')
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

