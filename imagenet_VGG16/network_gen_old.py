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

class VGG16(nn.Module):
	def __init__(self, init_weights=True):
		super(VGG16,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(25088, 4096, bias=False),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(4096, 4096, bias=False),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(4096, 1000, bias=False),
		)
		self._initialize_weights()

	def forward(self,x):
		global glob_gau
		global glob_blur
		if args.print == 1:
			npimg = np.array(x,dtype=float)
			npimg = npimg.squeeze(0)
			scipy.misc.toimage(npimg).save("img0.png")
		#Noise generation part
		if (glob_gau==0)&(glob_blur==0):
			#no noise
			pass

		elif (glob_blur == 0)&(glob_gau == 1):
			#gaussian noise add
			
			gau_kernel = torch.randn(x.size())*args.gau
			x = Variable(gau_kernel.cuda()) + x
			

		elif (glob_gau == 0)&(glob_blur == 1):
			#blur noise add
			blur_kernel_partial = torch.FloatTensor(utils.genblurkernel(args.blur))
			blur_kernel_partial = torch.matmul(blur_kernel_partial.unsqueeze(1),torch.transpose(blur_kernel_partial.unsqueeze(1),0,1))
			kernel_size = blur_kernel_partial.size()[0]
			zeros = torch.zeros(kernel_size,kernel_size)
			blur_kernel = torch.cat((blur_kernel_partial,zeros,zeros,
			zeros,blur_kernel_partial,zeros,
			zeros,zeros,blur_kernel_partial),0)
			blur_kernel = blur_kernel.view(3,3,kernel_size,kernel_size)
			blur_padding = int((blur_kernel_partial.size()[0]-1)/2)
			#x = torch.nn.functional.conv2d(x, weight=blur_kernel.cuda(), padding=blur_padding)
			x = torch.nn.functional.conv2d(x, weight=Variable(blur_kernel.cuda()), padding=blur_padding)

		elif (glob_gau == 1) & (glob_blur == 1):
			#both gaussian and blur noise added
			blur_kernel_partial = torch.FloatTensor(utils.genblurkernel(args.blur))
			blur_kernel_partial = torch.matmul(blur_kernel_partial.unsqueeze(1),torch.transpose(blur_kernel_partial.unsqueeze(1),0,1))
			kernel_size = blur_kernel_partial.size()[0]
			zeros = torch.zeros(kernel_size,kernel_size)
			blur_kernel = torch.cat((blur_kernel_partial,zeros,zeros,
			zeros,blur_kernel_partial,zeros,
			zeros,zeros,blur_kernel_partial),0)
			blur_kernel = blur_kernel.view(3,3,kernel_size,kernel_size)
			blur_padding = int((blur_kernel_partial.size()[0]-1)/2)
			x = torch.nn.functional.conv2d(x, weight=Variable(blur_kernel.cuda()), padding=blur_padding)
			gau_kernel = torch.randn(x.size())*args.gau
			x = Variable(gau_kernel.cuda()) + x
		else:
			print("Something is wrong in noise adding part")
			exit()
		if args.print == 1:
			npimg = np.array(x,dtype=float)
			npimg = npimg.squeeze(0)
			scipy.misc.toimage(npimg).save("img1.png")
			exit()
		fixed = 0
		if fixed:
			x = quant(x)
			x = roundmax(x)

		out = self.conv1(x)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv2(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv4(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv6(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv7(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv8(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv9(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv10(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv11(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv12(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv13(out)
		out = out.view(out.size(0), -1)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.fc1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.fc2(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.fc3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

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
		mask[13][:,:] = val 
		mask[14][:,:] = val 
		mask[15][:,:] = val 
	elif block == 1:
		for i in range(56):
			mask[0][i,:,:,:] = val
			mask[1][i,0:55,:,:] = val 
		for i in range(112):
			mask[2][i,0:55,:,:] = val 
			mask[3][i,0:111,:,:] = val
		for i in range(224):
			mask[4][i,0:111,:,:] = val
			mask[5][i,0:223,:,:] = val
			mask[6][i,0:223,:,:] = val
		for i in range(448):
			mask[7][i,0:223,:,:] = val
			mask[8][i,0:447,:,:] = val
			mask[9][i,0:447,:,:] = val
			mask[10][i,0:447,:,:] = val
			mask[11][i,0:447,:,:] = val
			mask[12][i,0:447,:,:] = val
		mask[13][0:3583,0:21951] = val 
		mask[14][0:3583,0:3583] = val 
		mask[15][:,0:3583] = val 
	elif block == 2:
		for i in range(48):
			mask[0][i,:,:,:] = val
			mask[1][i,0:47,:,:] = val 
		for i in range(96):
			mask[2][i,0:47,:,:] = val 
			mask[3][i,0:95,:,:] = val
		for i in range(192):
			mask[4][i,0:95,:,:] = val
			mask[5][i,0:191,:,:] = val
			mask[6][i,0:191,:,:] = val
		for i in range(384):
			mask[7][i,0:191,:,:] = val
			mask[8][i,0:383,:,:] = val
			mask[9][i,0:383,:,:] = val
			mask[10][i,0:383,:,:] = val
			mask[11][i,0:383,:,:] = val
			mask[12][i,0:383,:,:] = val
		mask[13][0:3071,0:18815] = val 
		mask[14][0:3071,0:3071] = val 
		mask[15][:,0:3071] = val 
	elif block == 3:
		for i in range(40):
			mask[0][i,:,:,:] = val
			mask[1][i,0:39,:,:] = val 
		for i in range(80):
			mask[2][i,0:39,:,:] = val 
			mask[3][i,0:79,:,:] = val
		for i in range(160):
			mask[4][i,0:79,:,:] = val
			mask[5][i,0:159,:,:] = val
			mask[6][i,0:159,:,:] = val
		for i in range(320):
			mask[7][i,0:159,:,:] = val
			mask[8][i,0:319,:,:] = val
			mask[9][i,0:319,:,:] = val
			mask[10][i,0:319,:,:] = val
			mask[11][i,0:319,:,:] = val
			mask[12][i,0:319,:,:] = val
		mask[13][0:2559,0:15679] = val 
		mask[14][0:2559,0:2559] = val 
		mask[15][:,0:2559] = val 
	elif block == 4:
		for i in range(32):
			mask[0][i,:,:,:] = val
			mask[1][i,0:31,:,:] = val 
		for i in range(64):
			mask[2][i,0:31,:,:] = val 
			mask[3][i,0:63,:,:] = val
		for i in range(128):
			mask[4][i,0:63,:,:] = val
			mask[5][i,0:127,:,:] = val
			mask[6][i,0:127,:,:] = val
		for i in range(256):
			mask[7][i,0:127,:,:] = val
			mask[8][i,0:255,:,:] = val
			mask[9][i,0:255,:,:] = val
			mask[10][i,0:255,:,:] = val
			mask[11][i,0:255,:,:] = val
			mask[12][i,0:255,:,:] = val
		mask[13][0:2047,0:12543] = val 
		mask[14][0:2047,0:2047] = val 
		mask[15][:,0:2047] = val 
	return mask

def net_mask_mul(net, mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.mul(param.data,mask[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.mul(param.data,mask[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.mul(param.data,mask[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.mul(param.data,mask[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.mul(param.data,mask[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.mul(param.data,mask[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.mul(param.data,mask[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.mul(param.data,mask[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.mul(param.data,mask[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.mul(param.data,mask[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.mul(param.data,mask[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.mul(param.data,mask[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.mul(param.data,mask[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.mul(param.data,mask[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.mul(param.data,mask[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.mul(param.data,mask[15])
	return net

def add_network(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,layer[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.add(param.data,layer[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.add(param.data,layer[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.add(param.data,layer[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.add(param.data,layer[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.add(param.data,layer[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.add(param.data,layer[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.add(param.data,layer[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.add(param.data,layer[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.add(param.data,layer[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.add(param.data,layer[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.add(param.data,layer[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.add(param.data,layer[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.add(param.data,layer[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.add(param.data,layer[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.add(param.data,layer[15])

def add_mask(net, mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,mask[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.add(param.data,mask[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.add(param.data,mask[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.add(param.data,mask[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.add(param.data,mask[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.add(param.data,mask[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.add(param.data,mask[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.add(param.data,mask[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.add(param.data,mask[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.add(param.data,mask[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.add(param.data,mask[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.add(param.data,mask[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.add(param.data,mask[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.add(param.data,mask[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.add(param.data,mask[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.add(param.data,mask[15])

def save_network(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			layer[0] = param.data
	for child in net.children():
		for param in child.conv2[0].parameters():
			layer[1] = param.data		
	for child in net.children():
		for param in child.conv3[0].parameters():
			layer[2] = param.data		
	for child in net.children():
		for param in child.conv4[0].parameters():
			layer[3] = param.data		
	for child in net.children():
		for param in child.conv5[0].parameters():
			layer[4] = param.data	
	for child in net.children():
		for param in child.conv6[0].parameters():
			layer[5] = param.data
	for child in net.children():
		for param in child.conv7[0].parameters():
			layer[6] = param.data
	for child in net.children():
		for param in child.conv8[0].parameters():
			layer[7] = param.data
	for child in net.children():
		for param in child.conv9[0].parameters():
			layer[8] = param.data
	for child in net.children():
		for param in child.conv10[0].parameters():
			layer[9] = param.data
	for child in net.children():
		for param in child.conv11[0].parameters():
			layer[10] = param.data
	for child in net.children():
		for param in child.conv12[0].parameters():
			layer[11] = param.data
	for child in net.children():
		for param in child.conv13[0].parameters():
			layer[12] = param.data

	for child in net.children():
		for param in child.fc1[1].parameters():
			layer[13] = param.data
	for child in net.children():
		for param in child.fc2[1].parameters():
			layer[14] = param.data
	for child in net.children():
		for param in child.fc3[0].parameters():
			layer[15] = param.data

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
	mask_rand = torch.load('mask_rand2.dat')
	layer = torch.load('mask_null.dat')
	try:
		checkpoint = torch.load('./checkpoint/'+args.block1)
		net1 = checkpoint['net']
		if use_cuda:
			net1.cuda() 
			net1 = torch.nn.DataParallel(net1, device_ids=range(0,8))
	except:
		print("Error : Failed to load net1. End program.")
		exit()

	try:
		if args.block2 == 'NULL':
			pass
		else:
			checkpoint = torch.load('./checkpoint/'+args.block2)
			net2 = checkpoint['net'] 
			net2.cuda() 
			net2 = torch.nn.DataParallel(net2, device_ids=range(0,8))
	except:
		print("Error : Failed to load net2. End program.")
		exit()

	try:
		if args.block3 == 'NULL':
			pass
		else:
			checkpoint = torch.load('./checkpoint/'+args.block3)
			net3 = checkpoint['net'] 
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
		for i in range(16):
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
		for i in range(16):
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
		for i in range(16):
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
		for i in range(16):
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
		for i in range(16):
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
	#'''
	f = open('testout.txt','a+')
	for child in net1.children():
		for param in child.conv2[0].parameters():
			for i in range(0,64):
				for j in range(0,64):
					print("data[{},{},:,:] = {}".format(i,j,param.data[i,j,:,:]), file=f)
	f.close()
	#'''

	if args.o == 'NULL':
		pass
	else:
		#torch.save(net1, './checkpoint/ckpt_20180613_half_clean_0.125_gaussian.t0')
		print('Saving..')
		state = {
			'net': net1.module if use_cuda else net1,
			'top1_acc': 0,
			'top5_acc': 0,
		}
	
		torch.save(state, './checkpoint/'+args.o)
	#printweight(net1)
