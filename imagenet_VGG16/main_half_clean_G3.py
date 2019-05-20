from __future__ import print_function

import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import argparse
import torch.optim as optim

import imagenet_custom_dataset as cd

from utils import progress_bar

import numpy as np
import utils
import os

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
parser.add_argument('--print', default=0, type=int, help='print input and dirty img to png') #mode=1 is train, mode=0 is inference
parser.add_argument('--prindex', default=0.0005, type=float)
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--gau', type=float, default=0, metavar='N',help='gaussian noise standard deviation')
parser.add_argument('--blur', type=float, default=0, metavar='N',help='blur noise standard deviation')
parser.add_argument('--network', default='ckpt_20190103_half_clean_G3.t0', help='input network ckpt name', metavar="FILE")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
top1_acc = 0  # best test accuracy
top5_acc = 0  # best test accuracy

traindir = os.path.join('/usr/share/ImageNet/train')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,num_workers=8, pin_memory=True)

valdir = os.path.join('/usr/share/ImageNet/val')
val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir,transforms.Compose([transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])),batch_size=128, shuffle=False,num_workers=8, pin_memory=True)

global glob_gau
global glob_blur
glob_gau = 0
glob_blur = 0

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

if args.mode == 0:
	checkpoint = torch.load('./checkpoint/'+args.network)
	net = checkpoint['net']

elif args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./checkpoint/ckpt_20190103_half_clean_G3.t0')
		net = checkpoint['net']
		ckpt = torch.load('./checkpoint/ckpt_20190103_half_clean_G2.t0')
		net2 = ckpt['net']
		top1_acc = checkpoint['top1_acc'] 
		top5_acc = checkpoint['top5_acc'] 
	else:
		checkpoint = torch.load('./checkpoint/ckpt_20190103_half_clean_G3.t0')
		net = checkpoint['net']
		ckpt = torch.load('./checkpoint/ckpt_20190103_half_clean_G2.t0')
		net2 = ckpt['net']
		top1_acc = 0
		top5_acc = 0

elif args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20190103_half_clean_G3.t0')
	net = checkpoint['net']
	#ckpt = torch.load('./checkpoint/ckpt_20180722_half_clean_prune_80_pprec_15.t0')
	#net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		top1_acc = checkpoint['top1_acc'] 
		top5_acc = checkpoint['top5_acc'] 
	else:
		top1_acc = 0
		top5_acc = 0

elif args.mode == 3:
	checkpoint = torch.load('./checkpoint/'+args.network)
	net = checkpoint['net']
	params = paramsget()
	thres = findThreshold(params)
	exit()

if use_cuda:
	#print(torch.cuda.device_count())
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

start_epoch = args.se
num_epoch = args.ne

###################################################################################
# Copied this part from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

######################################################################################

def train(epoch):
	global top1_acc
	global top5_acc
	global glob_gau
	global glob_blur
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	net.train()

	end = time.time()
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(set_mask(mask_channel, 0, 1), 2, 0)
	#mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		glob_gau = 0

		# measure data loading time
		data_time.update(time.time() - end)

		if use_cuda is not None:
			inputs, targets = inputs.cuda(), targets.cuda()

		# compute output
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1[0], inputs.size(0))
		top5.update(prec5[0], inputs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		net_mask_mul(mask_channel)
		add_network() 

		optimizer.step()

		if batch_idx % 200 == 0:
			batch_time.update(time.time() - end)
			end = time.time()
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   epoch, batch_idx, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		glob_gau = 1

		# measure data loading time
		data_time.update(time.time() - end)

		if use_cuda is not None:
			inputs, targets = inputs.cuda(), targets.cuda()

		# compute output
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1[0], inputs.size(0))
		top5.update(prec5[0], inputs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		net_mask_mul(mask_channel)
		add_network() 

		optimizer.step()

		# measure elapsed time
		if batch_idx % 200 == 0:
			batch_time.update(time.time() - end)
			end = time.time()
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   epoch, batch_idx, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))

def test():
	global top1_acc
	global top5_acc
	global glob_gau
	global glob_blur
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	net.eval()
	
	end = time.time()
	count = 0
	if args.mode == 1:
		mask_channel = torch.load('mask_null.dat')
		#mask_channel = set_mask(mask_channel, 4, 1)
		mask_channel = set_mask(set_mask(mask_channel, 0, 1), 2, 0)
		net_mask_mul(mask_channel)
		add_network()
	for batch_idx, (inputs, targets) in enumerate(val_loader):
		glob_gau = 1
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec1[0], inputs.size(0))
		top5.update(prec5[0], inputs.size(0))

		# measure elapsed time

		if batch_idx % 50 == 0:
			batch_time.update(time.time() - end)
			end = time.time()
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   batch_idx, len(val_loader), batch_time=batch_time, loss=losses,
				   top1=top1, top5=top5))

	# Save checkpoint.
	if top1.avg > top1_acc:
		if mode == 0:
			print('Acc : {}'.format(top1.avg))
			return
		else:
			print('Saving.. Acc : {}'.format(top1.avg))
			state = {
				'net': net.module if use_cuda else net,
				'top1_acc': top1.avg,
				'top5_acc': top5.avg,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt_20190103_half_clean_G3.t0')
			top1_acc = top1.avg

def retrain(epoch):
	global glob_gau
	global glob_blur
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	net.train()

	end = time.time()

	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(set_mask(mask_channel, 0, 1), 2, 0)
	#mask_channel = set_mask(mask_channel, 4, 1)

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if use_cuda is not None:
			inputs, targets = inputs.cuda(), targets.cuda()

		# compute output
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1[0], inputs.size(0))
		top5.update(prec5[0], inputs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		#quantize()

		net_mask_mul(mask_channel)
		#add_network()

		pruneNetwork(mask)

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if batch_idx % 200 == 0:
			batch_time.update(time.time() - end)
			end = time.time()
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   epoch, batch_idx, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))

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
	params = torch.cat((params,net.fc1[0].weight.view(-1,)),0)
	params = torch.cat((params,net.fc2[0].weight.view(-1,)),0)
	params = torch.cat((params,net.fc3[0].weight.view(-1,)),0)
	#net = checkpoint['net']
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)>thres).type(torch.FloatTensor)
		#result = torch.sum(tmp)/params.size()[0]*64/28
		#result = torch.sum(tmp)/params.size()[0]*64/11
		#result = torch.sum(tmp)/params.size()[0]*64/9
		result = torch.sum(tmp)/params.size()[0]*4 #for half clean
		#result = torch.sum(tmp)/params.size()[0] # for full size
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
	mask[13] = torch.abs(net.fc1[0].weight.data)>thres
	mask[14] = torch.abs(net.fc2[0].weight.data)>thres
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
		for param in child.fc1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[13].cuda())
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[14].cuda())
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[15].cuda())
			param.data = torch.mul(param.data,mask[15].cuda())
	return

def set_mask(mask, block, val):
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

def save_network(net):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = net.conv1[0].weight.data
		mask[1] = net.conv2[0].weight.data
		mask[2] = net.conv3[0].weight.data
		mask[3] = net.conv4[0].weight.data
		mask[4] = net.conv5[0].weight.data
		mask[5] = net.conv6[0].weight.data
		mask[6] = net.conv7[0].weight.data
		mask[7] = net.conv8[0].weight.data
		mask[8] = net.conv9[0].weight.data
		mask[9] = net.conv10[0].weight.data 
		mask[10] = net.conv11[0].weight.data
		mask[11] = net.conv12[0].weight.data
		mask[12] = net.conv13[0].weight.data
		mask[13] = net.fc1[0].weight.data
		mask[14] = net.fc2[0].weight.data
		mask[15] = net.fc3[0].weight.data
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = param.data
		for child in net.children():
			for param in child.conv2[0].parameters():
				mask[1] = param.data		
		for child in net.children():
			for param in child.conv3[0].parameters():
				mask[2] = param.data		
		for child in net.children():
			for param in child.conv4[0].parameters():
				mask[3] = param.data		
		for child in net.children():
			for param in child.conv5[0].parameters():
				mask[4] = param.data	
		for child in net.children():
			for param in child.conv6[0].parameters():
				mask[5] = param.data
		for child in net.children():
			for param in child.conv7[0].parameters():
				mask[6] = param.data
		for child in net.children():
			for param in child.conv8[0].parameters():
				mask[7] = param.data
		for child in net.children():
			for param in child.conv9[0].parameters():
				mask[8] = param.data
		for child in net.children():
			for param in child.conv10[0].parameters():
				mask[9] = param.data
		for child in net.children():
			for param in child.conv11[0].parameters():
				mask[10] = param.data
		for child in net.children():
			for param in child.conv12[0].parameters():
				mask[11] = param.data
		for child in net.children():
			for param in child.conv13[0].parameters():
				mask[12] = param.data

		for child in net.children():
			for param in child.fc1[0].parameters():
				mask[13] = param.data
		for child in net.children():
			for param in child.fc2[0].parameters():
				mask[14] = param.data
		for child in net.children():
			for param in child.fc3[0].parameters():
				mask[15] = param.data
	return mask

def add_network():
	layer = save_network(net2)
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
		for param in child.fc1[0].parameters():
			param.data = torch.add(param.data,layer[13])
	for child in net.children():
		for param in child.fc2[0].parameters():
			param.data = torch.add(param.data,layer[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.add(param.data,layer[15])

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
		for param in child.fc1[0].parameters():
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[0].parameters():
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.mul(param.data,mask[15].cuda())

def roundmax(input):
	maximum = 2 ** args.iwidth - 1
	minimum = -maximum - 1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum - minimum))
	input = torch.add(torch.neg(input), maximum)
	return input

def quant(input):
	input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return input

mode = args.mode
if mode == 0: # only inference
	test()
elif mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		print("epoch : {}".format(epoch))
		print(time.ctime())
		train(epoch)

		test()
elif mode == 2: # retrain for quantization and pruning
	for epoch in range(0,50):
		print("epoch : {}".format(epoch))
		print(time.ctime())
		retrain(epoch, mask_prune) 

		test()
else:
	pass
