from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import math
import torch.optim as optim

from utils import progress_bar

import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

traindir = os.path.join("/home/yhbyun/imagenet/")
valdir = os.path.join("/home/mhha/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transforms.Compose([transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,])),batch_size=100, shuffle=False,num_workers=8, pin_memory=True)

class RESNET34(nn.Module):
	def __init__(self):
		super(RESNET34,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)

		self.layer1_conv1 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_conv2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)

		self.layer1_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer1_conv3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_conv4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)

		self.layer1_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer1_conv5 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_conv6 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)

		self.layer1_relu6 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer2_conv1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_conv2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_downsample = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(128),
		)

		self.layer2_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer2_conv3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)

		self.layer2_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer2_conv5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_conv6 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)

		self.layer2_relu6 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer2_conv7 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_conv8 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)

		self.layer2_relu8 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv1 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_conv2 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_downsample = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv3 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_conv4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv5 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True)
		)
		self.layer3_conv6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu6 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_conv8 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu8 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv9 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_conv10 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu10 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_conv11 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_conv12 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)

		self.layer3_relu12 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer4_conv1 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_conv2 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)
		self.layer4_downsample = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(512),
		)

		self.layer4_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer4_conv3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_conv4 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)

		self.layer4_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer4_conv5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_conv6 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)

		self.layer4_relu6 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.avgpool = nn.Sequential(
			nn.AvgPool2d(7, stride=1, padding=0),
		)

		self.linear = nn.Sequential(
			nn.Linear(512, 1000)
		)

	def forward(self,x):
		out = self.conv1(x)

		residual = out
		out = self.layer1_conv1(out)
		out = self.layer1_conv2(out)
		out += residual
		out = self.layer1_relu2(out)

		residual = out
		out = self.layer1_conv3(out)
		out = self.layer1_conv4(out)
		out += residual
		out = self.layer1_relu4(out)

		residual = out
		out = self.layer1_conv5(out)
		out = self.layer1_conv6(out)
		out += residual
		out = self.layer1_relu6(out)

		residual = self.layer2_downsample(out)
		out = self.layer2_conv1(out)
		out = self.layer2_conv2(out)
		out += residual
		out = self.layer2_relu2(out)

		residual = out
		out = self.layer2_conv3(out)
		out = self.layer2_conv4(out)
		out += residual
		out = self.layer2_relu4(out)

		residual = out
		out = self.layer2_conv5(out)
		out = self.layer2_conv6(out)
		out += residual
		out = self.layer2_relu6(out)

		residual = out
		out = self.layer2_conv7(out)
		out = self.layer2_conv8(out)
		out += residual
		out = self.layer2_relu8(out)

		residual = self.layer3_downsample(out)
		out = self.layer3_conv1(out)
		out = self.layer3_conv2(out)
		out += residual
		out = self.layer3_relu2(out)

		residual = out
		out = self.layer3_conv3(out)
		out = self.layer3_conv4(out)
		out += residual
		out = self.layer3_relu4(out)

		residual = out
		out = self.layer3_conv5(out)
		out = self.layer3_conv6(out)
		out += residual
		out = self.layer3_relu6(out)

		residual = out
		out = self.layer3_conv7(out)
		out = self.layer3_conv8(out)
		out += residual
		out = self.layer3_relu8(out)

		residual = out
		out = self.layer3_conv9(out)
		out = self.layer3_conv10(out)
		out += residual
		out = self.layer3_relu10(out)

		residual = out
		out = self.layer3_conv11(out)
		out = self.layer3_conv12(out)
		out += residual
		out = self.layer3_relu12(out)

		residual = self.layer4_downsample(out)
		out = self.layer4_conv1(out)
		out = self.layer4_conv2(out)
		out += residual
		out = self.layer4_relu2(out)

		residual = out
		out = self.layer4_conv3(out)
		out = self.layer4_conv4(out)
		out += residual
		out = self.layer4_relu4(out)

		residual = out
		out = self.layer4_conv5(out)
		out = self.layer4_conv6(out)
		out += residual
		out = self.layer4_relu6(out)

		out = self.avgpool(out)

		out = out.view(out.size(0), -1)

		out = self.linear(out)

		return out


net = models.resnet34(pretrained=True)
if args.mode == 0:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180725.t0')
	net = checkpoint['net']

elif args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./checkpoint/ckpt_20180725.t0')
		best_acc = checkpoint['acc'] 
		net = checkpoint['net']
	else:
		print('==> Building model..')
		net = RESNET34()

elif args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20180725.t0')
	net = checkpoint['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0

if use_cuda:
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
		try:
			progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
		except:
			print('test_loss:',type(test_loss))

	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.module if use_cuda else net2,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		if args.mode == 0:
			pass
		else:
			torch.save(state, './checkpoint/ckpt_20180725.t0')
		best_acc = acc

# Train+inference vs. Inference
mode = args.mode
if mode == 0: # only inference
	test()
elif mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)

		test()
else:
	pass
'''elif mode == 2: # retrain for quantization and pruning
	for epoch in range(0,50):
		retrain(epoch, mask_prune) 

		test()
'''
