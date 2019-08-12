'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

import scipy.misc
from scipy import ndimage
import numpy as np

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 55.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

##########################################################################
# Codes under this line is written by YH.Byun

def saveInitialParameter(net):
	try:
		net_param = torch.load('mask_null.dat')
		net_param[0] = net.conv1[0].weight.data
		net_param[1] = net.conv2[0].weight.data
		net_param[2] = net.conv3[0].weight.data
		net_param[3] = net.conv4[0].weight.data
		net_param[4] = net.conv5[0].weight.data
		net_param[5] = net.conv6[0].weight.data
		net_param[6] = net.conv7[0].weight.data
		net_param[7] = net.conv8[0].weight.data
		net_param[8] = net.conv9[0].weight.data
		net_param[9] = net.conv10[0].weight.data 
		net_param[10] = net.conv11[0].weight.data
		net_param[11] = net.conv12[0].weight.data
		net_param[12] = net.conv13[0].weight.data
		net_param[13] = net.fc1[1].weight.data
		net_param[14] = net.fc2[1].weight.data
		net_param[15] = net.fc3[0].weight.data
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				net_param[0] = param.data
		for child in net.children():
			for param in child.conv2[0].parameters():
				net_param[1] = param.data
		for child in net.children():
			for param in child.conv3[0].parameters():
				net_param[2] = param.data
		for child in net.children():
			for param in child.conv4[0].parameters():
				net_param[3] = param.data
		for child in net.children():
			for param in child.conv5[0].parameters():
				net_param[4] = param.data
		for child in net.children():
			for param in child.conv6[0].parameters():
				net_param[5] = param.data
		for child in net.children():
			for param in child.conv7[0].parameters():
				net_param[6] = param.data
		for child in net.children():
			for param in child.conv8[0].parameters():
				net_param[7] = param.data
		for child in net.children():
			for param in child.conv9[0].parameters():
				net_param[8] = param.data
		for child in net.children():
			for param in child.conv10[0].parameters():
				net_param[9] = param.data
		for child in net.children():
			for param in child.conv11[0].parameters():
				net_param[10] = param.data
		for child in net.children():
			for param in child.conv12[0].parameters():
				net_param[11] = param.data
		for child in net.children():
			for param in child.conv13[0].parameters():
				net_param[12] = param.data
		for child in net.children():
			for param in child.fc1[1].parameters():
				net_param[13] = param.data
		for child in net.children():
			for param in child.fc2[1].parameters():
				net_param[14] = param.data
		for child in net.children():
			for param in child.fc3[0].parameters():
				net_param[15] = param.data
	torch.save(net_param,'./net_initialized_param.dat')
	print("saving initial parameters")
	
def quantize(net, pprec):
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
	return net

def maskGen(num_class):
	mask=[]

	mask_conv0 = torch.zeros(64,3,3,3).cuda()
	mask_conv1 = torch.zeros(64,64,3,3).cuda()

	mask_conv2 = torch.zeros(128,64,3,3).cuda()
	mask_conv3 = torch.zeros(128,128,3,3).cuda()

	mask_conv4 = torch.zeros(256,128,3,3).cuda()
	mask_conv5 = torch.zeros(256,256,3,3).cuda()
	mask_conv6 = torch.zeros(256,256,3,3).cuda()

	mask_conv7 = torch.zeros(512,256,3,3).cuda()
	mask_conv8 = torch.zeros(512,512,3,3).cuda()
	mask_conv9 = torch.zeros(512,512,3,3).cuda()
	mask_conv10 = torch.zeros(512,512,3,3).cuda()
	mask_conv11 = torch.zeros(512,512,3,3).cuda()
	mask_conv12 = torch.zeros(512,512,3,3).cuda()
	mask_fc0 = torch.zeros(512,512).cuda()
	mask_fc1 = torch.zeros(512,512).cuda()

	mask_fc2 = torch.zeros(num_class,512).cuda()

	mask.append(mask_conv0)
	mask.append(mask_conv1)
	mask.append(mask_conv2)
	mask.append(mask_conv3)
	mask.append(mask_conv4)
	mask.append(mask_conv5)
	mask.append(mask_conv6)
	mask.append(mask_conv7)
	mask.append(mask_conv8)
	mask.append(mask_conv9)
	mask.append(mask_conv10)
	mask.append(mask_conv11)
	mask.append(mask_conv12)
	mask.append(mask_fc0)
	mask.append(mask_fc1)
	mask.append(mask_fc2)

	torch.save(mask, 'mask_null.dat')

def pruneNetwork(net, mask):
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
	return net

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

def findThreshold(net, params, pr):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)<thres).type(torch.FloatTensor)
		result = torch.sum(tmp)/params.size()[0]
		if (pr/100)<result:
			#print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(net, thres):
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

def netMaskMul(net, mask):
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

