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

from utils import progress_bar

import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--fixed', default=0, type=int, help='quantization') #mode=1 is train, mode=0 is inference

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

#traindir = os.path.join("/home/yhbyun/imagenet/")
traindir = os.path.join('/home/yhbyun/Imagenet2010/','train')
valdir = os.path.join("/home/mhha/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=200, shuffle=False,
		num_workers=4, pin_memory=True)

class VGG16(nn.Module):
	def __init__(self):
		super(VGG16,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.linear1 = nn.Sequential(
			nn.Linear(25088, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
		)
		self.linear2 = nn.Sequential(
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
		)
		self.linear3 = nn.Sequential(
			nn.Linear(4096, 1000),
		)

	def forward(self,x):
		if args.fixed:
			x = quant(x)
			x = roundmax(x)

		out = self.conv1(x)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv2(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv4(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv5(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv6(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv7(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv8(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv9(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv10(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv11(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv12(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.conv13(out)
		out = out.view(out.size(0), -1)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear1(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear2(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear3(out)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		return out

if args.mode == 0:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180726.t0')
	net = checkpoint['net']

elif args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./checkpoint/ckpt_20180726.t0')
		best_acc = checkpoint['acc'] 
		net = checkpoint['net']
	else:
		print('==> Building model..')
		net = VGG16()

elif args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20180726.t0')
	net = checkpoint['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0

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
	global best_acc
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	net.train()

	end = time.time()
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
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		#progress_bar(batch_idx, len(train_loader), 'Loss: {loss.val:.4f} | Acc: %.3f%% (%d/%d)'
		#	% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		progress_bar(batch_idx, len(train_loader), 
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))
	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net2.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt_20180726.t0')
		best_acc = acc

def test():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(val_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		test_loss += loss.data[0].item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
	'''
	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net2.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt_20180726.t0')
		best_acc = acc
	'''

def roundmax(input):
	'''
	maximum = 2 ** args.iwidth - 1
	minimum = -maximum - 1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum - minimum))
	input = torch.add(torch.neg(input), maximum)
	'''
	return input

def quant(input):
	#input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return input

mode = args.mode
if mode == 0: # only inference
	test()
elif mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)

		test()
		print("epoch : {}".format(epoch))
else:
	pass
