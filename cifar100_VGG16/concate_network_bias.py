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

use_cuda = torch.cuda.is_available()

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,3,padding=1,bias=True), #layer0
			nn.BatchNorm2d(64), # batch norm is added because dataset is changed
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3,padding=1, bias=True), #layer3
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.maxpool1 = nn.Sequential(
			nn.MaxPool2d(2,2), # 16*16* 64
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64,128,3,padding=1, bias=True), #layer7
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128,128,3,padding=1, bias=True),#layer10
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.maxpool2 = nn.Sequential(
			nn.MaxPool2d(2,2), # 8*8*128
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128,256,3,padding=1, bias=True), #layer14
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=True), #layer17
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=True), #layer20
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.maxpool3 = nn.Sequential(
			nn.MaxPool2d(2,2), # 4*4*256
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256,512,3,padding=1, bias=True), #layer24
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=True), #layer27
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=True), #layer30
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool4 = nn.Sequential(
			nn.MaxPool2d(2,2), # 2*2*512
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=True), #layer34
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=True), #layer37
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=True), #layer40
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool5 = nn.Sequential(
			nn.MaxPool2d(2,2) # 1*1*512
		)
		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=True), #fc_layer1
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=True), #fc_layer4
			nn.ReLU(inplace=True),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(512,100, bias=True) #fc_layer6
		)

	def forward(self,x):
		isfixed = 0
		if isfixed:
			x = quant(x)
			x = roundmax(x)

		out1 = self.conv1(x) # 1250*64*32*32
		
		if isfixed:
			out1 = quant(out1) 
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if isfixed:
			out2 = quant(out2)
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)
		out4 = self.conv3(out3) # 1250*128*16*16
		if isfixed:
			out4 = quant(out4) 
			out4 = roundmax(out4)
		out5 = self.conv4(out4) # 1250*128*16*16
		if isfixed:
			out5 = quant(out5) 
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)
		out7 = self.conv5(out6) # 1250*256*8*8
		if isfixed:
			out7 = quant(out7) 
			out7 = roundmax(out7)
		out8 = self.conv6(out7) # 1250*256*8*8
		if isfixed:
			out8 = quant(out8) 
			out8 = roundmax(out8)
		out9 = self.conv7(out8) # 1250*256*8*8
		if isfixed:
			out9 = quant(out9) 
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)
		out11 = self.conv8(out10) # 1250*512*4*4
		if isfixed:
			out11 = quant(out11) 
			out11 = roundmax(out11)
		out12 = self.conv9(out11) # 1250*512*4*4
		if isfixed:
			out12 = quant(out12) 
			out12 = roundmax(out12)
		out13 = self.conv10(out12) # 1250*512*4*
		if isfixed:
			out13 = quant(out13) 
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*
		if isfixed:
			out15 = quant(out15) 
			out15 = roundmax(out15)
		out16 = self.conv12(out15) # 1250*512*2*
		if isfixed:
			out16 = quant(out16) 
			out16 = roundmax(out16)
		out17 = self.conv13(out16) # 1250*512*2*
		if isfixed:
			out17 = quant(out17) 
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)
		out20 = self.fc1(out19) # 1250*512
		if isfixed:
			out20 = quant(out20) 
			out20 = roundmax(out20)
		out21 = self.fc2(out20) # 1250*512
		if isfixed:
			out21 = quant(out21) 
			out21 = roundmax(out21)
		out22 = self.fc3(out21) # 1250*10
		if isfixed:
			out22 = quant(out22) 
			out22 = roundmax(out22)

		return out22

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

def set_mask(mask, block, val):
	if block == 0:
		#weigth masking
		mask[0][:,:,:,:] = val
		mask[2][:,:,:,:] = val 
		mask[4][:,:,:,:] = val 
		mask[6][:,:,:,:] = val
		mask[8][:,:,:,:] = val
		mask[10][:,:,:,:] = val
		mask[12][:,:,:,:] = val
		mask[14][:,:,:,:] = val
		mask[16][:,:,:,:] = val
		mask[18][:,:,:,:] = val
		mask[20][:,:,:,:] = val
		mask[22][:,:,:,:] = val
		mask[24][:,:,:,:] = val
		mask[26][:,:] = val 
		mask[28][:,:] = val 
		mask[30][:,:] = val 

		#bias masking
		mask[1][:] = val
		mask[3][:] = val 
		mask[5][:] = val 
		mask[7][:] = val
		mask[9][:] = val
		mask[11][:] = val
		mask[13][:] = val
		mask[15][:] = val
		mask[17][:] = val
		mask[19][:] = val
		mask[21][:] = val
		mask[23][:] = val
		mask[25][:] = val
		mask[27][:] = val 
		mask[29][:] = val 
	elif block == 1:
		for i in range(56):
			mask[0][i,:,:,:] = val
			mask[2][i,0:55,:,:] = val 
		for i in range(112):
			mask[4][i,0:55,:,:] = val 
			mask[6][i,0:111,:,:] = val
		for i in range(224):
			mask[8][i,0:111,:,:] = val
			mask[10][i,0:223,:,:] = val
			mask[12][i,0:223,:,:] = val
		for i in range(448):
			mask[14][i,0:223,:,:] = val
			mask[16][i,0:447,:,:] = val
			mask[18][i,0:447,:,:] = val
			mask[20][i,0:447,:,:] = val
			mask[22][i,0:447,:,:] = val
			mask[24][i,0:447,:,:] = val
			mask[26][i,0:447] = val 
			mask[28][i,0:447] = val 
		mask[30][:,0:447] = val 

		#bias masking
		mask[1][56] = val
		mask[3][56] = val 
		mask[5][112] = val 
		mask[7][112] = val
		mask[9][224] = val
		mask[11][224] = val
		mask[13][224] = val
		mask[15][448] = val
		mask[17][448] = val
		mask[19][448] = val
		mask[21][448] = val
		mask[23][448] = val
		mask[25][448] = val
		mask[27][448] = val 
		mask[29][448] = val 
	elif block == 2:
		for i in range(48):
			mask[0][i,:,:,:] = val
			mask[2][i,0:47,:,:] = val 
		for i in range(96):
			mask[4][i,0:47,:,:] = val 
			mask[6][i,0:95,:,:] = val
		for i in range(192):
			mask[8][i,0:95,:,:] = val
			mask[10][i,0:191,:,:] = val
			mask[12][i,0:191,:,:] = val
		for i in range(384):
			mask[14][i,0:191,:,:] = val
			mask[16][i,0:383,:,:] = val
			mask[18][i,0:383,:,:] = val
			mask[20][i,0:383,:,:] = val
			mask[22][i,0:383,:,:] = val
			mask[24][i,0:383,:,:] = val
			mask[26][i,0:383] = val 
			mask[28][i,0:383] = val 
		mask[30][:,0:383] = val 

		#bias masking
		mask[1][48] = val
		mask[3][48] = val 
		mask[5][96] = val 
		mask[7][96] = val
		mask[9][192] = val
		mask[11][192] = val
		mask[13][192] = val
		mask[15][384] = val
		mask[17][384] = val
		mask[19][384] = val
		mask[21][384] = val
		mask[23][384] = val
		mask[25][384] = val
		mask[27][384] = val 
		mask[29][384] = val 

	elif block == 3:
		for i in range(40):
			mask[0][i,:,:,:] = val
			mask[2][i,0:39,:,:] = val 
		for i in range(80):
			mask[4][i,0:39,:,:] = val 
			mask[6][i,0:79,:,:] = val
		for i in range(160):
			mask[8][i,0:79,:,:] = val
			mask[10][i,0:159,:,:] = val
			mask[12][i,0:159,:,:] = val
		for i in range(320):
			mask[14][i,0:159,:,:] = val
			mask[16][i,0:319,:,:] = val
			mask[18][i,0:319,:,:] = val
			mask[20][i,0:319,:,:] = val
			mask[22][i,0:319,:,:] = val
			mask[24][i,0:319,:,:] = val
			mask[26][i,0:319] = val 
			mask[28][i,0:319] = val 
		mask[30][:,0:319] = val 

		#bias masking
		mask[1][40] = val
		mask[3][40] = val 
		mask[5][80] = val 
		mask[7][80] = val
		mask[9][160] = val
		mask[11][160] = val
		mask[13][160] = val
		mask[15][320] = val
		mask[17][320] = val
		mask[19][320] = val
		mask[21][320] = val
		mask[23][320] = val
		mask[25][320] = val
		mask[27][320] = val 
		mask[29][320] = val 

	elif block == 4:
		for i in range(32):
			mask[0][i,:,:,:] = val
			mask[2][i,0:31,:,:] = val 
		for i in range(64):
			mask[4][i,0:31,:,:] = val 
			mask[6][i,0:63,:,:] = val
		for i in range(128):
			mask[8][i,0:63,:,:] = val
			mask[10][i,0:127,:,:] = val
			mask[12][i,0:127,:,:] = val
		for i in range(256):
			mask[14][i,0:127,:,:] = val
			mask[16][i,0:255,:,:] = val
			mask[18][i,0:255,:,:] = val
			mask[20][i,0:255,:,:] = val
			mask[22][i,0:255,:,:] = val
			mask[24][i,0:255,:,:] = val
			mask[26][i,0:255] = val 
			mask[28][i,0:255] = val 
		mask[30][:,0:255] = val 
		
		#bias masking
		mask[1][32] = val
		mask[3][32] = val 
		mask[5][64] = val 
		mask[7][64] = val
		mask[9][128] = val
		mask[11][128] = val
		mask[13][128] = val
		mask[15][256] = val
		mask[17][256] = val
		mask[19][256] = val
		mask[21][256] = val
		mask[23][256] = val
		mask[25][256] = val
		mask[27][256] = val 
		mask[29][256] = val 
	return mask

def net_mask_mul(net, mask):
	for child in net.children():
		for param, i in zip(child.conv1[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[0].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param, i in zip(child.conv2[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[2].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param, i in zip(child.conv3[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[4].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param, i in zip(child.conv4[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[6].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param, i in zip(child.conv5[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[8].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param, i in zip(child.conv6[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[10].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param, i in zip(child.conv7[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[12].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param, i in zip(child.conv8[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[14].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[15].cuda())
	for child in net.children():
		for param, i in zip(child.conv9[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[16].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[17].cuda())
	for child in net.children():
		for param, i in zip(child.conv10[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[18].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[19].cuda())
	for child in net.children():
		for param, i in zip(child.conv11[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[20].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[21].cuda())
	for child in net.children():
		for param, i in zip(child.conv12[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[22].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[23].cuda())
	for child in net.children():
		for param, i in zip(child.conv13[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[24].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[25].cuda())
	for child in net.children():
		for param, i in zip(child.fc1[1].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[26].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[27].cuda())
	for child in net.children():
		for param, i in zip(child.fc2[1].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[28].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[29].cuda())
	for child in net.children():
		for param, i in zip(child.fc3[0].parameters(), range(0,2)):
			if i==0:
				param.data = torch.mul(param.data,mask[30].cuda())
			if i==1:
				param.data = torch.mul(param.data,mask[31].cuda())
	return net

def printweight(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			f = open('test_1.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv2[0].parameters():
			f = open('test_2.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv3[0].parameters():
			f = open('test_3.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 3 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv4[0].parameters():
			f = open('test_4.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 4 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv5[0].parameters():
			f = open('test_5.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 5 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv6[0].parameters():
			f = open('test_6.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 6 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv7[0].parameters():
			f = open('test_7.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 7 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv8[0].parameters():
			f = open('test_8.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 8 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv9[0].parameters():
			f = open('test_9.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 9 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv10[0].parameters():
			f = open('test_10.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 10 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv11[0].parameters():
			f = open('test_11.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 11 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv12[0].parameters():
			f = open('test_12.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 12 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv13[0].parameters():
			f = open('test_13.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 13 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc1[1].parameters():
			f = open('test_fc1.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc2[1].parameters():
			f = open('test_fc2.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc3[0].parameters():
			f = open('test_fc3.csv','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 3 weight printed')
			f.close()

if __name__ == '__main__':
	if use_cuda:
		cudnn.benchmark = True

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
	
	net_result = CNN()
	set_block(3,1)
	net1 = block_network(net1)
