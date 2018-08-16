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
top1_acc = 0  # best test accuracy
top5_acc = 0  # best test accuracy

#traindir = os.path.join("/home/yhbyun/imagenet/")
traindir = os.path.join('/data/ImageNet2012/','train')
#valdir = os.path.join('/home/yhbyun/Imagenet2010/','val')
#valdset = cd.IMAGENET2010VAL("/home/yhbyun/examples/imagenet/ILSVRC2010_validation_ground_truth.csv")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))

#train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])),batch_size=args.bs, shuffle=False,num_workers=8, pin_memory=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,num_workers=8, pin_memory=True)

#'''
valdir = os.path.join("/home/mhha/", 'val')
val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir,transforms.Compose([transforms.Scale(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])),batch_size=1, shuffle=False,num_workers=4, pin_memory=True)
#'''
'''
valdir = os.path.join("/home/mhha/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=80, shuffle=False,
        num_workers=4, pin_memory=True)
'''

#val_loader = torch.utils.data.DataLoader(valdset, batch_size=1, shuffle=False,	num_workers=8, pin_memory=True)

class VGG16(nn.Module):
	def __init__(self, init_weights=True):
		super(VGG16,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.linear1 = nn.Sequential(
			nn.Linear(25088, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.linear2 = nn.Sequential(
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.linear3 = nn.Sequential(
			nn.Linear(4096, 1000),
		)
		self._initialize_weights()

	def forward(self,x):
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

		out = self.linear1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear2(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				#print(m)
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				#print(m)
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
				'''
				f = open('fcparam.txt','w')
				for param in m.parameters():
					print(param.data)
				#print(m.parameters())
				#print(m.data.cpu(), file=f)
				f.close()
				'''

if args.mode == 0:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt_20180813.t0')
	#checkpoint = torch.load('./checkpoint/backup_ckpt_20180813.t0')
	net = checkpoint['net']

	f = open('fcparam.txt','w')
	for i in range(0,1000):
		print(net.linear3[0].weight[i,0].cpu(), file = f)
	f.close()
	exit()

elif args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./checkpoint/ckpt_20180813.t0')
		top1_acc = checkpoint['top1_acc'] 
		top5_acc = checkpoint['top5_acc'] 
		net = checkpoint['net']
	else:
		print('==> Building model..')
		net = VGG16()
		exit()
		top1_acc = 0
		top5_acc = 0

elif args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_20180813.t0')
	net = checkpoint['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		top1_acc = checkpoint['top1_acc'] 
		top5_acc = checkpoint['top5_acc'] 
	else:
		top1_acc = 0
		top5_acc = 0

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

		#progress_bar(batch_idx, len(train_loader), 'Loss: {loss.val:.4f} | Acc: %.3f%% (%d/%d)'
		#	% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
		'''
		progress_bar(batch_idx, len(train_loader), 
				  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   batch_time=batch_time,loss=losses, top1=top1, top5=top5))
		'''
	# Save checkpoint.
	print('Saving..')
	state = {
		'net': net.module if use_cuda else net,
		'top1_acc': 0,
		'top5_acc': 0,
	}
	torch.save(state, './checkpoint/ckpt_20180813_'+str(top1.avg)+'.t0')
	top1_acc =0 

'''
def test():
	global top1_acc
	global top5_acc
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

	# Save checkpoint.
	acc = 100.*correct/total
	if acc > top1_acc:
		print('Saving..')
		state = {
			'net': net.module if use_cuda else net,
			'top1_acc': top1_acc,
			'top5_acc': top5_acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt_20180813.t0')
		best_acc = top1_acc
'''

def test():
	global top1_acc
	global top5_acc
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	net.eval()
	
	end = time.time()
	count = 0
	for batch_idx, (inputs, targets) in enumerate(val_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
		'''
		if count == 100:
			"""
			for i in range(0,80):
				print((outputs[i].cpu()==torch.max(outputs[i].cpu())).nonzero())
				print(targets[i].cpu())
				print('')
			exit()
			"""
			print(outputs[0].cpu())
		else:
			count += 1
		'''
		#print((outputs.cpu()==torch.max(outputs.cpu())).nonzero())
		#print((targets.cpu()==torch.max(targets.cpu())).nonzero())
		
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec1[0], inputs.size(0))
		top5.update(prec5[0], inputs.size(0))

		# measure elapsed time

		'''
		test_loss += loss.data[0].item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		

		progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
		'''

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
		
		'''
		progress_bar(batch_idx, len(val_loader), 
				  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   batch_time=batch_time,loss=losses, top1=top1, top5=top5))
		'''

	'''
	# Save checkpoint.
	if top1.avg > top1_acc:
		if mode == 0:
			return
		else:
			print('Saving..')
			state = {
				'net': net.module if use_cuda else net,
				'top1_acc': top1.avg,
				'top5_acc': top5.avg,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt_20180813.t0')
			top1_acc = top1.avg
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
		#print(time.ctime())
		train(epoch)

		test()
		print("epoch : {}".format(epoch))
else:
	pass
