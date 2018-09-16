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

import struct
import random

parser = argparse.ArgumentParser(description='load and make new network')
parser.add_argument('--mode', default=0, type=int, help='mode 1 -> for 0.125, mode 2 -> for 0.25, mode 3 -> for full channel')
parser.add_argument('--block1', default='NULL', help='input block1 ckpt name', metavar="FILE")
parser.add_argument('--block2', default='NULL', help='input block2 ckpt name', metavar="FILE")
parser.add_argument('--block3', default='NULL', help='input block3 ckpt name', metavar="FILE")
parser.add_argument('--o', default='NULL', help='output file name', metavar="FILE")


use_cuda = torch.cuda.is_available()
args = parser.parse_args()

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
	'''maximum = 2**iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)'''
	return input	

def quant(input):
	#input = torch.round(input / (2 ** (-aprec))) * (2 ** (-aprec))
	return input


def set_mask(block, val):
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

def add_network(net):
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

def net_mask_mul(net, mask):
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
	return net

def add_mask(net, mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,mask[0])
	for child in net.children():
		for param in child.layer1_basic1[0].parameters():
			param.data = torch.add(param.data,mask[1])		
	for child in net.children():
		for param in child.layer1_basic2[0].parameters():
			param.data = torch.add(param.data,mask[2])		
	for child in net.children():
		for param in child.layer1_basic3[0].parameters():
			param.data = torch.add(param.data,mask[3])		
	for child in net.children():
		for param in child.layer1_basic4[0].parameters():
			param.data = torch.add(param.data,mask[4])	
	for child in net.children():
		for param in child.layer2_basic1[0].parameters():
			param.data = torch.add(param.data,mask[5])
	for child in net.children():
		for param in child.layer2_basic2[0].parameters():
			param.data = torch.add(param.data,mask[6])
	for child in net.children():
		for param in child.layer2_basic3[0].parameters():
			param.data = torch.add(param.data,mask[7])
	for child in net.children():
		for param in child.layer2_basic4[0].parameters():
			param.data = torch.add(param.data,mask[8])
	for child in net.children():
		for param in child.layer3_basic1[0].parameters():
			param.data = torch.add(param.data,mask[9])
	for child in net.children():
		for param in child.layer3_basic2[0].parameters():
			param.data = torch.add(param.data,mask[10])
	for child in net.children():
		for param in child.layer3_basic3[0].parameters():
			param.data = torch.add(param.data,mask[11])
	for child in net.children():
		for param in child.layer3_basic4[0].parameters():
			param.data = torch.add(param.data,mask[12])
	for child in net.children():
		for param in child.layer4_basic1[0].parameters():
			param.data = torch.add(param.data,mask[13])
	for child in net.children():
		for param in child.layer4_basic2[0].parameters():
			param.data = torch.add(param.data,mask[14])
	for child in net.children():
		for param in child.layer4_basic3[0].parameters():
			param.data = torch.add(param.data,mask[15])
	for child in net.children():
		for param in child.layer4_basic4[0].parameters():
			param.data = torch.add(param.data,mask[16])
	for child in net.children():
		for param in child.layer2_downsample[0].parameters():
			param.data = torch.add(param.data,mask[17])
	for child in net.children():
		for param in child.layer3_downsample[0].parameters():
			param.data = torch.add(param.data,mask[18])
	for child in net.children():
		for param in child.layer4_downsample[0].parameters():
			param.data = torch.add(param.data,mask[19])
	for child in net.children():
		for param in child.linear[0].parameters():
			param.data = torch.add(param.data,mask[20])

def printweight(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			f = open('test_1.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv2[0].parameters():
			f = open('test_2.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv3[0].parameters():
			f = open('test_3.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 3 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv4[0].parameters():
			f = open('test_4.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 4 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv5[0].parameters():
			f = open('test_5.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 5 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv6[0].parameters():
			f = open('test_6.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 6 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv7[0].parameters():
			f = open('test_7.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 7 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv8[0].parameters():
			f = open('test_8.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 8 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv9[0].parameters():
			f = open('test_9.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 9 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv10[0].parameters():
			f = open('test_10.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 10 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv11[0].parameters():
			f = open('test_11.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 11 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv12[0].parameters():
			f = open('test_12.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 12 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv13[0].parameters():
			f = open('test_13.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 13 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc1[1].parameters():
			f = open('test_fc1.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc2[1].parameters():
			f = open('test_fc2.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc3[0].parameters():
			f = open('test_fc3.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 3 weight printed')
			f.close()

if __name__ == '__main__':
	if use_cuda:
		cudnn.benchmark = True

	mask = torch.load('mask_null.dat')
	mask_rand = torch.load('mask_rand.dat')
	layer = torch.load('mask_null.dat')
	try:
		checkpoint = torch.load('./checkpoint/'+args.block1)
		net1 = checkpoint['net']
		if use_cuda:
			net1.cuda() 
			net1 = torch.nn.DataParallel(net1, device_ids=range(0,8))
	except Exception as e:
		print("Error : Failed to load net1. End program.")
		print("type error : " + str(e))
		exit()

	try:
		if args.block2 == 'NULL':
			checkpoint = torch.load('./checkpoint/ckpt_null.t0')
			net2 = checkpoint['net']
		else:
			checkpoint = torch.load('./checkpoint/'+args.block2)
			net2 = checkpoint['net'] 
		if use_cuda:
			net2.cuda() 
			net2 = torch.nn.DataParallel(net2, device_ids=range(0,8))
	except:
		print("Error : Failed to load net2. End program.")
		exit()

	try:
		if args.block3 == 'NULL':
			checkpoint = torch.load('./checkpoint/ckpt_null.t0')
			net3 = checkpoint['net']
		else:
			checkpoint = torch.load('./checkpoint/'+args.block3)
			net3 = checkpoint['net'] 
		if use_cuda:
			net3.cuda() 
			net3 = torch.nn.DataParallel(net3, device_ids=range(0,8))
	except:
		print("Error : Failed to load net3. End program.")
		exit()
	
	#######################################################
	#Enable this part for blur 06, gau 008
	#'''
	if args.mode == 1:
		mask = set_mask(3,1)
		mask = set_mask(4,0)
		for i in range(21):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 06, gau 008 threshold check
	#'''
	if args.mode == 4:
		mask = set_mask(3,1)
		mask = set_mask(4,0)
		net1 = net_mask_mul(net1,mask)
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 08, gau 016
	#'''
	if args.mode == 2:
		mask = set_mask(3,1)
		#print(type(mask))
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(2,1)
		mask = set_mask(3,0)
		for i in range(21):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 08, gau 016 threshold check
	#'''
	if args.mode == 5:
		mask = set_mask(2,1)
		mask = set_mask(3,0)
		net1 = net_mask_mul(net1,mask)
	#'''
	#######################################################
	
	#######################################################
	#Enable this part for blur 10, gau 025
	#'''	
	if args.mode == 3:
		mask = set_mask(2,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(0,1)
		mask = set_mask(2,0)
		for i in range(21):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 10, gau 025 threshold check
	#'''
	if args.mode == 6:
		mask = set_mask(0,1)
		mask = set_mask(2,0)
		net1 = net_mask_mul(net1,mask)
	#'''
	#######################################################

	#######################################################
	#'''	
	if args.mode == 7:
		mask = set_mask(2,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(1,1)
		mask = set_mask(2,0)
		for i in range(21):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#'''	
	if args.mode == 8:
		mask = set_mask(1,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(0,1)
		mask = set_mask(1,0)
		for i in range(21):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################
	
	#######################################################
	#Enable this part for gaussian 016
	'''
	mask = set_mask(3,1)
	net1 = net_mask_mul(net1)
	mask = set_mask(0,0)
	mask = set_mask(2,1)
	mask = set_mask(3,0)
	for i in range(16):
		mask[i] = torch.mul(mask[i],mask_rand[i])
	add_mask(net1,mask) 
	'''
	#######################################################

	#######################################################
	#Enable this part for gaussian 025
	'''
	mask = set_mask(2,1)
	net1 = net_mask_mul(net1)
	mask = set_mask(0,1)
	mask = set_mask(2,0)
	for i in range(16):
		mask[i] = torch.mul(mask[i],mask_rand[i])
	add_mask(net1,mask) 
	'''
	#######################################################
	
	#######################################################
	#Check if training works
	'''
	f = open('testout1.csv','a+')
	for child in net1.children():
		for param in child.conv10[0].parameters():
			print(torch.sum(torch.abs(param)), file=f)
	f.close()

	mask = set_mask(0,1)
	mask = set_mask(4,0)
	net1 = net_mask_mul(net1)
	#net2 = net_mask_mul(net2)
	#save_network(net2)
	#net1 = add_network(net1)

	f = open('testout2.csv','a+')
	for child in net1.children():
		for param in child.conv10[0].parameters():
			print(torch.sum(torch.abs(param)),file=f)
	f.close()
	'''
	#######################################################
	'''
	f = open('testout.txt','a+')
	for child in net1.children():
		for param in child.conv2[0].parameters():
			for i in range(0,64):
				for j in range(0,64):
					print("data[{},{},:,:] = {}".format(i,j,param.data[i,j,:,:]), file=f)
	f.close()
	'''

	if args.o == 'NULL':
		pass
	else:
		#torch.save(net1, './checkpoint/ckpt_20180613_half_clean_0.125_gaussian.t0')
		print('Saving..')
		state = {
			'net': net1.module if use_cuda else net1,
			'acc': 0,
		}
	
		torch.save(state, './checkpoint/'+args.o)
	#printweight(net1)
