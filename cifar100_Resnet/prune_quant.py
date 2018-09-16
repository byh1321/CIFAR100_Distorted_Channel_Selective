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
#import Resnet_vision2 as RS
import Resnet34 as RS

import struct
import random

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')

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

def paramsget():
	params = net.conv1[0].weight.view(-1,)
	params = torch.cat((params,net.layer1_basic1[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer1_basic2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer1_basic3[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer1_basic4[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer1_basic5[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer1_basic6[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic1[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic3[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic4[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic5[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic6[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic7[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_basic8[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic1[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic3[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic4[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic5[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic6[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic7[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic8[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic9[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic10[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic11[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_basic12[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic1[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic3[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic4[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic5[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_basic6[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer2_downsample[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer3_downsample[0].weight.view(-1,)),0)
	params = torch.cat((params,net.layer4_downsample[0].weight.view(-1,)),0)
	params = torch.cat((params,net.linear[0].weight.view(-1,)),0)
	#net = checkpoint['net']
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)<thres).type(torch.FloatTensor)
		result = torch.sum(tmp)/params.size()[0]
		if (args.pr/100)<result:
			print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(thres):
	mask = torch.load('mask_null.dat')
	mask[0] = torch.abs(net.layer1_basic1[0].weight.data)>thres
	mask[1] = torch.abs(net.layer1_basic2[0].weight.data)>thres
	mask[2] = torch.abs(net.layer1_basic3[0].weight.data)>thres
	mask[3] = torch.abs(net.layer1_basic4[0].weight.data)>thres
	mask[4] = torch.abs(net.layer1_basic5[0].weight.data)>thres
	mask[5] = torch.abs(net.layer1_basic6[0].weight.data)>thres
	mask[6] = torch.abs(net.layer2_basic1[0].weight.data)>thres
	mask[7] = torch.abs(net.layer2_basic2[0].weight.data)>thres
	mask[8] = torch.abs(net.layer2_basic3[0].weight.data)>thres
	mask[9] = torch.abs(net.layer2_basic4[0].weight.data)>thres
	mask[10] = torch.abs(net.layer2_basic5[0].weight.data)>thres
	mask[11] = torch.abs(net.layer2_basic6[0].weight.data)>thres
	mask[12] = torch.abs(net.layer2_basic7[0].weight.data)>thres
	mask[13] = torch.abs(net.layer2_basic8[0].weight.data)>thres
	mask[14] = torch.abs(net.layer3_basic1[0].weight.data)>thres
	mask[15] = torch.abs(net.layer3_basic2[0].weight.data)>thres
	mask[16] = torch.abs(net.layer3_basic3[0].weight.data)>thres
	mask[17] = torch.abs(net.layer3_basic4[0].weight.data)>thres
	mask[18] = torch.abs(net.layer3_basic5[0].weight.data)>thres
	mask[19] = torch.abs(net.layer3_basic6[0].weight.data)>thres
	mask[20] = torch.abs(net.layer3_basic7[0].weight.data)>thres
	mask[21] = torch.abs(net.layer3_basic8[0].weight.data)>thres
	mask[22] = torch.abs(net.layer3_basic9[0].weight.data)>thres
	mask[23] = torch.abs(net.layer3_basic10[0].weight.data)>thres
	mask[24] = torch.abs(net.layer3_basic11[0].weight.data)>thres
	mask[25] = torch.abs(net.layer3_basic12[0].weight.data)>thres
	mask[26] = torch.abs(net.layer4_basic1[0].weight.data)>thres
	mask[27] = torch.abs(net.layer4_basic2[0].weight.data)>thres
	mask[28] = torch.abs(net.layer4_basic3[0].weight.data)>thres
	mask[29] = torch.abs(net.layer4_basic4[0].weight.data)>thres
	mask[30] = torch.abs(net.layer4_basic5[0].weight.data)>thres
	mask[31] = torch.abs(net.layer4_basic6[0].weight.data)>thres
	mask[32] = torch.abs(net.layer2_downsample[0].weight.data)>thres
	mask[33] = torch.abs(net.layer3_downsample[0].weight.data)>thres
	mask[34] = torch.abs(net.layer4_downsample[0].weight.data)>thres
	mask[0] = mask[0].type(torch.FloatTensor)
	mask[1] = mask[1].type(torch.FloatTensor)
	mask[2] = mask[2].type(torch.FloatTensor)
	mask[3] = mask[3].type(torch.FloatTensor)
	mask[4] = mask[4].type(torch.FloatTensor)
	mask[5] = mask[5].type(torch.FloatTensor)
	mask[6] = mask[6].type(torch.FloatTensor)
	mask[7] = mask[7].type(torch.FloatTensor)
	mask[8] = mask[8].type(torch.FloatTensor)
	mask[9] = mask[9].type(torch.FloatTensor)
	mask[10] = mask[10].type(torch.FloatTensor)
	mask[11] = mask[11].type(torch.FloatTensor)
	mask[12] = mask[12].type(torch.FloatTensor)
	mask[13] = mask[13].type(torch.FloatTensor)
	mask[14] = mask[14].type(torch.FloatTensor)
	mask[15] = mask[15].type(torch.FloatTensor)
	mask[16] = mask[16].type(torch.FloatTensor)
	mask[17] = mask[17].type(torch.FloatTensor)
	mask[18] = mask[18].type(torch.FloatTensor)
	mask[19] = mask[19].type(torch.FloatTensor)
	mask[20] = mask[20].type(torch.FloatTensor)
	mask[21] = mask[21].type(torch.FloatTensor)
	mask[22] = mask[22].type(torch.FloatTensor)
	mask[23] = mask[23].type(torch.FloatTensor)
	mask[24] = mask[24].type(torch.FloatTensor)
	mask[25] = mask[25].type(torch.FloatTensor)
	mask[26] = mask[26].type(torch.FloatTensor)
	mask[27] = mask[27].type(torch.FloatTensor)
	mask[28] = mask[28].type(torch.FloatTensor)
	mask[29] = mask[29].type(torch.FloatTensor)
	mask[30] = mask[30].type(torch.FloatTensor)
	mask[31] = mask[31].type(torch.FloatTensor)
	mask[32] = mask[32].type(torch.FloatTensor)
	mask[33] = mask[33].type(torch.FloatTensor)
	mask[34] = mask[34].type(torch.FloatTensor)
	return mask


def pruneNetwork(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[0].cuda())
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[1].cuda())
			param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[2].cuda())
			param.data = torch.mul(param.data,mask[2].cuda())
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[3].cuda())
			param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[4].cuda())
			param.data = torch.mul(param.data,mask[4].cuda())
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[5].cuda())
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[6].cuda())
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[7].cuda())
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[8].cuda())
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[9].cuda())
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[10].cuda())
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[11].cuda())
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[12].cuda())
			param.data = torch.mul(param.data,mask[12].cuda())

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[13].cuda())
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[14].cuda())
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[15].cuda())
			param.data = torch.mul(param.data,mask[15].cuda())
	return

# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180723.t0')
	best_acc = 0 
	net = checkpoint['net']

else:
	print('==> Building model..')
	net = RS.ResNet34()

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
		torch.save(state, './checkpoint/ckpt_20180723.t0')
		best_acc = acc
	
	return acc

# Truncate weight param
'''
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
'''
#print(net)
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

