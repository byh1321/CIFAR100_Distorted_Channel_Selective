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
# import VGG16

import struct
import random
import math

def paramsget(net):
	try:
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
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				params = param.view(-1,)
		for child in net.children():
			for param in child.conv2[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv3[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv4[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv5[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv6[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv7[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv8[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv9[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv10[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv11[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv12[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.conv13[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)

		for child in net.children():
			for param in child.fc1[1].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.fc2[1].parameters():
				params = torch.cat((params,param.view(-1,)),0)
		for child in net.children():
			for param in child.fc3[0].parameters():
				params = torch.cat((params,param.view(-1,)),0)
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)<thres).type(torch.FloatTensor)
		result = torch.sum(tmp)/params.size()[0]
		if (args.pr/100)<result:
			#print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(thres):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = (torch.abs(net.conv1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[1] = (torch.abs(net.conv2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[2] = (torch.abs(net.conv3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[3] = (torch.abs(net.conv4[0].weight.data)>thres).type(torch.FloatTensor)
		mask[4] = (torch.abs(net.conv5[0].weight.data)>thres).type(torch.FloatTensor)
		mask[5] = (torch.abs(net.conv6[0].weight.data)>thres).type(torch.FloatTensor)
		mask[6] = (torch.abs(net.conv7[0].weight.data)>thres).type(torch.FloatTensor)
		mask[7] = (torch.abs(net.conv8[0].weight.data)>thres).type(torch.FloatTensor)
		mask[8] = (torch.abs(net.conv9[0].weight.data)>thres).type(torch.FloatTensor)
		mask[9] = (torch.abs(net.conv10[0].weight.data)>thres).type(torch.FloatTensor)
		mask[10] = (torch.abs(net.conv11[0].weight.data)>thres).type(torch.FloatTensor)
		mask[11] = (torch.abs(net.conv12[0].weight.data)>thres).type(torch.FloatTensor)
		mask[12] = (torch.abs(net.conv13[0].weight.data)>thres).type(torch.FloatTensor)
		mask[13] = (torch.abs(net.fc1[1].weight.data)>thres).type(torch.FloatTensor)
		mask[14] = (torch.abs(net.fc2[1].weight.data)>thres).type(torch.FloatTensor)
		mask[15] = (torch.abs(net.fc3[0].weight.data)>thres).type(torch.FloatTensor)
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv2[0].parameters():
				mask[1] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv3[0].parameters():
				mask[2] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv4[0].parameters():
				mask[3] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv5[0].parameters():
				mask[4] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv6[0].parameters():
				mask[5] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv7[0].parameters():
				mask[6] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv8[0].parameters():
				mask[7] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv9[0].parameters():
				mask[8] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv10[0].parameters():
				mask[9] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv11[0].parameters():
				mask[10] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv12[0].parameters():
				mask[11] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.conv13[0].parameters():
				mask[12] = (torch.abs(param.data)>thres).type(torch.FloatTensor)

		for child in net.children():
			for param in child.fc1[1].parameters():
				mask[13] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.fc2[1].parameters():
				mask[14] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
		for child in net.children():
			for param in child.fc3[0].parameters():
				mask[15] = (torch.abs(param.data)>thres).type(torch.FloatTensor)
	return mask

def getZeroPoints(net):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = (net.conv1[0].weight.data == 0).type(torch.FloatTensor)
		mask[1] = (net.conv2[0].weight.data == 0).type(torch.FloatTensor)
		mask[2] = (net.conv3[0].weight.data == 0).type(torch.FloatTensor)
		mask[3] = (net.conv4[0].weight.data == 0).type(torch.FloatTensor)
		mask[4] = (net.conv5[0].weight.data == 0).type(torch.FloatTensor)
		mask[5] = (net.conv6[0].weight.data == 0).type(torch.FloatTensor)
		mask[6] = (net.conv7[0].weight.data == 0).type(torch.FloatTensor)
		mask[7] = (net.conv8[0].weight.data == 0).type(torch.FloatTensor)
		mask[8] = (net.conv9[0].weight.data == 0).type(torch.FloatTensor)
		mask[9] = (net.conv10[0].weight.data == 0).type(torch.FloatTensor)
		mask[10] = (net.conv11[0].weight.data == 0).type(torch.FloatTensor)
		mask[11] = (net.conv12[0].weight.data == 0).type(torch.FloatTensor)
		mask[12] = (net.conv13[0].weight.data == 0).type(torch.FloatTensor)
		mask[13] = (net.fc1[1].weight.data == 0).type(torch.FloatTensor)
		mask[14] = (net.fc2[1].weight.data == 0).type(torch.FloatTensor)
		mask[15] = (net.fc3[0].weight.data == 0).type(torch.FloatTensor)
	return mask
	
def getNonZeroPoints(net):
	mask = torch.load('mask_null.dat')
	mask[0] = (net.conv1[0].weight.data != 0).type(torch.FloatTensor)
	mask[1] = (net.conv2[0].weight.data != 0).type(torch.FloatTensor)
	mask[2] = (net.conv3[0].weight.data != 0).type(torch.FloatTensor)
	mask[3] = (net.conv4[0].weight.data != 0).type(torch.FloatTensor)
	mask[4] = (net.conv5[0].weight.data != 0).type(torch.FloatTensor)
	mask[5] = (net.conv6[0].weight.data != 0).type(torch.FloatTensor)
	mask[6] = (net.conv7[0].weight.data != 0).type(torch.FloatTensor)
	mask[7] = (net.conv8[0].weight.data != 0).type(torch.FloatTensor)
	mask[8] = (net.conv9[0].weight.data != 0).type(torch.FloatTensor)
	mask[9] = (net.conv10[0].weight.data != 0).type(torch.FloatTensor)
	mask[10] = (net.conv11[0].weight.data != 0).type(torch.FloatTensor)
	mask[11] = (net.conv12[0].weight.data != 0).type(torch.FloatTensor)
	mask[12] = (net.conv13[0].weight.data != 0).type(torch.FloatTensor)
	mask[13] = (net.fc1[1].weight.data != 0).type(torch.FloatTensor)
	mask[14] = (net.fc2[1].weight.data != 0).type(torch.FloatTensor)
	mask[15] = (net.fc3[0].weight.data != 0).type(torch.FloatTensor)
	return mask

def findGPThreshold(params1, params2):
	thres=0
	while 1:
		tmp1 = (torch.abs(params1.data)>thres).type(torch.FloatTensor)
		tmp2 = (torch.abs(params2.data)>thres).type(torch.FloatTensor)
		result = torch.sum(tmp1)/params1.size()[0] - torch.sum(tmp2)/params1.size()[0]
		#print('thres : {}, result : {}, tmp : {}'.format(thres,result, torch.sum(tmp)))
		if ((100-args.pr)/100)>result.item():
			print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.00001

def net_mask_mul(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.mul(param.data,mask[2].cuda())
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.mul(param.data,mask[4].cuda())
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.mul(param.data,mask[12].cuda())

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.mul(param.data,mask[15].cuda())
	return net

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

def add_network():
	layer = torch.load('mask_null.dat')
	layer = save_network(layer)
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

def save_network(layer):
	for child in net2.children():
		for param in child.conv1[0].parameters():
			layer[0] = param.data
	for child in net2.children():
		for param in child.conv2[0].parameters():
			layer[1] = param.data		
	for child in net2.children():
		for param in child.conv3[0].parameters():
			layer[2] = param.data		
	for child in net2.children():
		for param in child.conv4[0].parameters():
			layer[3] = param.data		
	for child in net2.children():
		for param in child.conv5[0].parameters():
			layer[4] = param.data	
	for child in net2.children():
		for param in child.conv6[0].parameters():
			layer[5] = param.data
	for child in net2.children():
		for param in child.conv7[0].parameters():
			layer[6] = param.data
	for child in net2.children():
		for param in child.conv8[0].parameters():
			layer[7] = param.data
	for child in net2.children():
		for param in child.conv9[0].parameters():
			layer[8] = param.data
	for child in net2.children():
		for param in child.conv10[0].parameters():
			layer[9] = param.data
	for child in net2.children():
		for param in child.conv11[0].parameters():
			layer[10] = param.data
	for child in net2.children():
		for param in child.conv12[0].parameters():
			layer[11] = param.data
	for child in net2.children():
		for param in child.conv13[0].parameters():
			layer[12] = param.data

	for child in net2.children():
		for param in child.fc1[1].parameters():
			layer[13] = param.data
	for child in net2.children():
		for param in child.fc2[1].parameters():
			layer[14] = param.data
	for child in net2.children():
		for param in child.fc3[0].parameters():
			layer[15] = param.data
	return layer

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

