from __future__ import print_function

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

traindir = os.path.join("/home/yhbyun/imagenet/")
valdir = os.path.join("/home/mhha/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=1, shuffle=False,
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
	print(torch.cuda.device_count())
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

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
	for batch_idx, (inputs, targets) in enumerate(val_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		test_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

		progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

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
		train(epoch)

		test()
else:
	pass
