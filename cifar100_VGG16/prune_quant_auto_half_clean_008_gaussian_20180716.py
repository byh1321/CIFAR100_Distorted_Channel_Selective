'''
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
'''
############################################################################################
# example : python3 -W ignore prune_quant_auto_half_clean.py --mode 1 --bs 1024 --lr 0.8 --ne 100 
# 		  : python3 -W ignore prune_quant_auto_half_clean.py --mode 1 --bs 1024 --lr 0.08 --ne 100 -r 
#		  : python3 -W ignore prune_quant_auto_half_clean.py --mode 1 --bs 1024 --lr 0.008 --ne 100 -r
#		  : python3 -W ignore prune_quant_auto_half_clean.py --mode 1 --bs 128 --lr 0.001 --ne 100 -r
############################################################################################

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

import struct
import random
import cifar_dirty_test
import cifar_dirty_train
import concate_network as cn
#import VGG16_yh 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--thres', default=0, type=float)
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
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
cifar_test_blur_08 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.0_blur_0.8_test_targets.csv")
cifar_test_blur_06 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.6_test_targets.csv")
cifar_test_blur_03 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.3_test_targets.csv")

cifar_train_blur_10 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_1.0_train_targets.csv")
cifar_train_blur_08 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.0_blur_0.8_train_targets.csv")
cifar_train_blur_06 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.6_train_targets.csv")
cifar_train_blur_03 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_0.3_train_targets.csv")

cifar_train_gaussian_025 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_0.0_train_targets.csv")
cifar_train_blur_10 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.0_blur_1.0_train_targets.csv")

cifar_train_gaussian_008_blur_03_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.08_blur_0.3_train_targets.csv") 
cifar_train_gaussian_016_blur_06_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.16_blur_0.6_train_targets.csv") 
cifar_train_gaussian_016_blur_08_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/A2S/cifar100_VGG16/cifar100_gaussian_0.16_blur_0.8_train_targets.csv") 
cifar_train_gaussian_025_blur_10_mixed = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/180614_cifar_VGG16/cifar100_gaussian_0.25_blur_1.0_train_targets.csv") 

train_loader = torch.utils.data.DataLoader(cifar_train_gaussian_008,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
#train_loader = torch.utils.data.DataLoader(cifar_train_blur_03,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test_gaussian_008,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

mode = args.mode

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,3,padding=1,bias=False), #layer0
			nn.BatchNorm2d(64), # batch norm is added because dataset is changed
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3,padding=1, bias=False), #layer3
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.maxpool1 = nn.Sequential(
			nn.MaxPool2d(2,2), # 16*16* 64
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64,128,3,padding=1, bias=False), #layer7
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128,128,3,padding=1, bias=False),#layer10
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.maxpool2 = nn.Sequential(
			nn.MaxPool2d(2,2), # 8*8*128
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128,256,3,padding=1, bias=False), #layer14
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer17
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer20
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.maxpool3 = nn.Sequential(
			nn.MaxPool2d(2,2), # 4*4*256
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256,512,3,padding=1, bias=False), #layer24
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer27
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer30
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool4 = nn.Sequential(
			nn.MaxPool2d(2,2), # 2*2*512
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer34
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer37
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer40
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool5 = nn.Sequential(
			nn.MaxPool2d(2,2) # 1*1*512
		)
		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer1
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer4
			nn.ReLU(inplace=True),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(512,100, bias=False) #fc_layer6
		)

	def forward(self,x):
		if args.fixed:
			x = roundmax(x)
			x = quant(x)
		out1 = self.conv1(x) # 1250*64*32*32
		if args.fixed:
			out1 = quant(out1) 
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if args.fixed:
			out2 = quant(out2)
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)
		out4 = self.conv3(out3) # 1250*128*16*16
		if args.fixed:
			out4 = quant(out4) 
			out4 = roundmax(out4)
		out5 = self.conv4(out4) # 1250*128*16*16
		if args.fixed:
			out5 = quant(out5) 
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)
		out7 = self.conv5(out6) # 1250*256*8*8
		if args.fixed:
			out7 = quant(out7) 
			out7 = roundmax(out7)
		out8 = self.conv6(out7) # 1250*256*8*8
		if args.fixed:
			out8 = quant(out8) 
			out8 = roundmax(out8)
		out9 = self.conv7(out8) # 1250*256*8*8
		if args.fixed:
			out9 = quant(out9) 
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)
		out11 = self.conv8(out10) # 1250*512*4*4
		if args.fixed:
			out11 = quant(out11) 
			out11 = roundmax(out11)
		out12 = self.conv9(out11) # 1250*512*4*4
		if args.fixed:
			out12 = quant(out12) 
			out12 = roundmax(out12)
		out13 = self.conv10(out12) # 1250*512*4*
		if args.fixed:
			out13 = quant(out13) 
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*
		if args.fixed:
			out15 = quant(out15) 
			out15 = roundmax(out15)
		out16 = self.conv12(out15) # 1250*512*2*
		if args.fixed:
			out16 = quant(out16) 
			out16 = roundmax(out16)
		out17 = self.conv13(out16) # 1250*512*2*
		if args.fixed:
			out17 = quant(out17) 
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)
		out20 = self.fc1(out19) # 1250*512
		if args.fixed:
			out20 = quant(out20) 
			out20 = roundmax(out20)
		out21 = self.fc2(out20) # 1250*512
		if args.fixed:
			out21 = quant(out21) 
			out21 = roundmax(out21)
		out22 = self.fc3(out21) # 1250*10
		if args.fixed:
			out22 = quant(out22) 
			out22 = roundmax(out22)

		return out22

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

def paramsget():
	params = net.conv1[0].weight.view(-1,)
	params = torch.cat((params,net.conv2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv3[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv4[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv5[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv6[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv7[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv8[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv9[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv10[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv11[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv12[0].weight.view(-1,)),0)
	params = torch.cat((params,net.conv13[0].weight.view(-1,)),0)
	params = torch.cat((params,net.fc1[1].weight.view(-1,)),0)
	params = torch.cat((params,net.fc2[1].weight.view(-1,)),0)
	params = torch.cat((params,net.fc3[0].weight.view(-1,)),0)
	#net = checkpoint['net']
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)>thres).type(torch.FloatTensor)
		result = torch.sum(tmp)/params.size()[0]*4
		if ((100-args.pr)/100)>result:
			print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(thres):
	mask = torch.load('mask_null.dat')
	mask[0] = torch.abs(net.conv1[0].weight.data)>thres
	mask[1] = torch.abs(net.conv2[0].weight.data)>thres
	mask[2] = torch.abs(net.conv3[0].weight.data)>thres
	mask[3] = torch.abs(net.conv4[0].weight.data)>thres
	mask[4] = torch.abs(net.conv5[0].weight.data)>thres
	mask[5] = torch.abs(net.conv6[0].weight.data)>thres
	mask[6] = torch.abs(net.conv7[0].weight.data)>thres
	mask[7] = torch.abs(net.conv8[0].weight.data)>thres
	mask[8] = torch.abs(net.conv9[0].weight.data)>thres
	mask[9] = torch.abs(net.conv10[0].weight.data)>thres
	mask[10] = torch.abs(net.conv11[0].weight.data)>thres
	mask[11] = torch.abs(net.conv12[0].weight.data)>thres
	mask[12] = torch.abs(net.conv13[0].weight.data)>thres
	mask[13] = torch.abs(net.fc1[1].weight.data)>thres
	mask[14] = torch.abs(net.fc2[1].weight.data)>thres
	mask[15] = torch.abs(net.fc3[0].weight.data)>thres
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

# Load checkpoint.

if args.mode == 0:
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./checkpoint/'+args.network)
		net = checkpoint['net']

elif args.mode == 1:
	checkpoint = torch.load('./checkpoint/ckpt_20180716_half_clean_008_gaussian_prune_80_pprec_15_69.t0')
	ckpt = torch.load('./checkpoint/ckpt_20180716_half_clean_prune_80_pprec_15_69.t0')
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0

elif args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20180716_half_clean_008_gaussian_prune_80_pprec_15_69.t0')
	ckpt = torch.load('./checkpoint/ckpt_20180716_half_clean_prune_80_pprec_15_69.t0')
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0

if args.pr:
	params = paramsget()
	if args.thres == 0:
		thres = findThreshold(params)
	else:
		thres = args.thres
	mask_prune = getPruningMask(thres)

#printsize()
#net = checkpoint['net']
if use_cuda:
	net.cuda()
	net2.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

#pruneNetwork(mask)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	net2.train()
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
	net2.eval()
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
		state = {
			'net': net.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		if args.mode == 0:
			pass
		else:
			print('Saving..')
			torch.save(state,'./checkpoint/ckpt_20180716_half_clean_008_gaussian_prune_80_pprec_15_69.t0')
		best_acc = acc

	return acc
	
# Retraining
def retrain(epoch,mask):
	print('\nEpoch: %d' % epoch)
	global best_acc
	net.train()
	net2.train()
	train_loss = 0
	total = 0
	correct = 0

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		#quantize()

		pruneNetwork(mask)

		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		acc = 100.*correct/total

# Truncate weight param
pprec = args.pprec
def quantize():
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

mask_channel = torch.load('mask_null.dat')

# Train+inference vs. Inference
if mode == 0: # only inference
	test()

elif mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)
		
		mask_channel = cn.set_mask(cn.set_mask(mask_channel, 3, 1), 4, 0)
		net = cn.net_mask_mul(net, mask_channel)
		net = cn.add_network(net, net2) 

		test()
elif mode == 2: # retrain for quantization and pruning
	for epoch in range(0,20):
		retrain(epoch, mask_prune) 

		mask_channel = cn.set_mask(cn.set_mask(mask_channel, 3, 1), 4, 0)
		net = cn.net_mask_mul(net, mask_channel)
		net = cn.add_network(net, net2) 

		test()
else:
	pass

