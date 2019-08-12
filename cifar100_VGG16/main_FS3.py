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
import pytorch_fft.fft as fft
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse
# import VGG16
import cifar_dirty_test
import cifar_dirty_train

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
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--count', type=int, default=50000, metavar='N',help='number of test image')
parser.add_argument('--dirty', type=int, default=0, metavar='N',help='dirty dataset -> dirty = 1')
parser.add_argument('--testsel', type=int, default=7, metavar='N',help='choose testset')
parser.add_argument('--trainsel', type=int, default=7, metavar='N',help='choose trainset')
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

cifar_train = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=False, transform=transform_test, target_transform=None, download=True)

cifar_test_G1 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_noise_level1_test_targets.csv")
cifar_test_G2 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_noise_level2_test_targets.csv")
cifar_test_G3 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_noise_level3_test_targets.csv")
cifar_train_G1 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_noise_level1_train_targets.csv")
cifar_train_G2 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_noise_level2_train_targets.csv")
cifar_train_G3 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_noise_level3_train_targets.csv")

cifar_test_B1 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_blur_level1_test_targets.csv")
cifar_test_B2 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_blur_level2_test_targets.csv")
cifar_test_B3 = cifar_dirty_test.CIFAR100DIRTY_TEST("/home/yhbyun/Dataset/mhha/cifar100_blur_level3_test_targets.csv")
cifar_train_B1 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_blur_level1_train_targets.csv")
cifar_train_B2 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_blur_level2_train_targets.csv")
cifar_train_B3 = cifar_dirty_train.CIFAR100DIRTY_TRAIN("/home/yhbyun/Dataset/mhha/cifar100_blur_level3_train_targets.csv")


train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=1, shuffle=False,num_workers=8,drop_last=False)

if args.trainsel == 0:
	test_loader = torch.utils.data.DataLoader(cifar_train_B3,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 1:
	test_loader = torch.utils.data.DataLoader(cifar_train_B2,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 2:
	test_loader = torch.utils.data.DataLoader(cifar_train_B1,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 3:
	test_loader = torch.utils.data.DataLoader(cifar_train,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 4:
	test_loader = torch.utils.data.DataLoader(cifar_train_G1,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 5:
	test_loader = torch.utils.data.DataLoader(cifar_train_G2,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.trainsel == 6:
	test_loader = torch.utils.data.DataLoader(cifar_train_G3,batch_size=1, shuffle=False,num_workers=8,drop_last=False)

if args.testsel == 0:
	test_loader = torch.utils.data.DataLoader(cifar_test_B3,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 1:
	test_loader = torch.utils.data.DataLoader(cifar_test_B2,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 2:
	test_loader = torch.utils.data.DataLoader(cifar_test_B1,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 3:
	test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 4:
	test_loader = torch.utils.data.DataLoader(cifar_test_G1,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 5:
	test_loader = torch.utils.data.DataLoader(cifar_test_G2,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
elif args.testsel == 6:
	test_loader = torch.utils.data.DataLoader(cifar_test_G3,batch_size=1, shuffle=False,num_workers=8,drop_last=False)

mode = args.mode

def roundmax(input):
	maximum = 2**args.iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	return input	

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
		x = quant(x)
		x = roundmax(x)
		tmp = Variable(torch.zeros(1,3,32,32).cuda())
		f = fft.Fft2d()
		fft_rout, fft_iout = f(x, tmp)
		mag = torch.sqrt(torch.mul(fft_rout,fft_rout) + torch.mul(fft_iout,fft_iout))
		tmp = torch.zeros(1,1,32,32).cuda()
		tmp = torch.add(torch.add(mag[:,0,:,:],mag[:,1,:,:]),mag[:,2,:,:])
		tmp = torch.abs(tmp)
		PFSUM = 0
		'''
		for i in range(0,32):
			for j in range(0,32):
				if (i+j) > 15:
					print_value = 0
				elif (i-j) < 17:
					print_value = 0
				elif (j-i) < 17:
					print_value = 0
				elif (i+j) < 48:
					print_value = 0
				else:
					PFSUM = PFSUM + tmp[0,i,j]
		'''
		freq_l = 0
		freq_h = 0
		for i in range(0,32):
			for j in range(0,32):
				#'''
				if (i+j) < 11:
					freq_l = freq_l + torch.abs(tmp[0,i,j])
				elif (i-j) > 21:
					freq_l = freq_l + torch.abs(tmp[0,i,j])
				elif (j-i) > 21:
					freq_l = freq_l + torch.abs(tmp[0,i,j])
				elif (i+j) > 52:
					freq_l = freq_l + torch.abs(tmp[0,i,j])
				else:
					freq_h = freq_h + torch.abs(tmp[0,i,j])
		PFSUM = freq_l - freq_h
		f = open(args.outputfile,'a+')
		print(PFSUM.item(),file=f)
		f.close()
		'''
		f = open('clean_train_freq.txt','w')
		for i in range(0,32):
			for j in range(0,32):
				print(tmp.data[0,i,j]/3,file = f)
		f.close()
		'''
		#print(mag.size())
		#print(tmp.size())
		#####################################
		# print original image
		'''
		f = open('Fixedimage.txt','a+')
		for i in range(0,32):
			for j in range(0,32):
				tmp2 = torch.add(torch.add(x[0,0,i,j],x[0,1,i,j]),x[0,2,i,j])
				print(tmp2.data[0]/3,file = f, end='\t')
				#print(mag[0,1,i,j].data[0],file = f)
				#print(mag[0,2,i,j].data[0],file = f)
			print('',file=f)
		f.close()
		'''
		#####################################

		#####################################
		# print fft image
		'''
		f = open('Fixedfft.txt','a+')
		for i in range(0,mag.size()[2]):
			for j in range(0,mag.size()[3]):
				print(tmp[0,i,j].data[0]/3,file = f, end='\t')
				#print(mag[0,1,i,j].data[0],file = f)
				#print(mag[0,2,i,j].data[0],file = f)
			print('',file=f)
		f.close()
		'''
		#####################################

		'''
		if args.fixed:
			x = roundmax(x)
		out1 = self.conv1(x) # 1250*64*32*32
		if args.fixed:
			out1 = torch.round(out1 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if args.fixed:
			out2 = torch.round(out2 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)

		out4 = self.conv3(out3) # 1250*128*16*16
		if args.fixed:
			out4 = torch.round(out4 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out4 = roundmax(out4)

		out5 = self.conv4(out4) # 1250*128*16*16
		if args.fixed:
			out5 = torch.round(out5 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)

		out7 = self.conv5(out6) # 1250*256*8*8
		if args.fixed:
			out7 = torch.round(out7 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out7 = roundmax(out7)

		out8 = self.conv6(out7) # 1250*256*8*8
		if args.fixed:
			out8 = torch.round(out8 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out8 = roundmax(out8)

		out9 = self.conv7(out8) # 1250*256*8*8
		if args.fixed:
			out9 = torch.round(out9 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)

		out11 = self.conv8(out10) # 1250*512*4*4
		if args.fixed:
			out11 = torch.round(out11 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out11 = roundmax(out11)

		out12 = self.conv9(out11) # 1250*512*4*4
		if args.fixed:
			out12 = torch.round(out12 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out12 = roundmax(out12)

		out13 = self.conv10(out12) # 1250*512*4*4
		if args.fixed:
			out13 = torch.round(out13 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*2
		if args.fixed:
			out15 = torch.round(out15 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out15 = roundmax(out15)

		out16 = self.conv12(out15) # 1250*512*2*2
		if args.fixed:
			out16 = torch.round(out16 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out16 = roundmax(out16)

		out17 = self.conv13(out16) # 1250*512*2*2
		if args.fixed:
			out17 = torch.round(out17 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)

		out20 = self.fc1(out19) # 1250*512
		if args.fixed:
			out20 = torch.round(out20 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out20 = roundmax(out20)

		out21 = self.fc2(out20) # 1250*512
		if args.fixed:
			out21 = torch.round(out21 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out21 = roundmax(out21)

		out22 = self.fc3(out21) # 1250*10
		if args.fixed:
			out22 = torch.round(out22 / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
			out22 = roundmax(out22)

		return out22
		'''

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

# Model
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20190802_half_clean.t0')
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
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		'''
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		#progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		#	% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		'''
		net(inputs)
		progress_bar(batch_idx, len(test_loader))
	exit()



def test():
	global best_acc
	global count
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	count = 0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		'''
		if count < args.count:
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.data[0]
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()

			progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
			f = open('result_original.txt','a+')
			print('{:.5f}'.format(100. * correct / len(test_loader.dataset)), end='\t', file=f)
			f.close()
		else:
			acc=0
			retur'''
		net(inputs)
		progress_bar(batch_idx, len(test_loader))
	exit()
		#print(len(test_loader))
		#count = count + 1


	# Save checkpoint.
	'''if count<args.count:
		acc = 100.*correct/total
		if acc > best_acc:
			print('Saving..')
			state = {
				'net': net.module if use_cuda else net,
				'acc': acc,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			#torch.save(state, './checkpoint/ckpt_prune70.t1')
			best_acc = acc'''
	
	#return acc

mode = args.mode
# Train+inference vs. Inference
if mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)
		test()
else:
	test()

print('\n')
