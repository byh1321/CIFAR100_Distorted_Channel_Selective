"""
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import progress_bar

import scipy.misc
from scipy import ndimage

import os
import numpy as np
import argparse

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gaussian_std', default=0, type=float, help='std of gaussian noise')
parser.add_argument('--blur_std', default=0, type=float, help='std of blur noise')
parser.add_argument('--test', default=0, type=float, help='make test dataset if test=1')
parser.add_argument('--train', default=0, type=float, help='make train dataset if train=1')

args = parser.parse_args()

transform_train = transforms.Compose([transforms.ToTensor(),
					  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
transform_test = transforms.Compose([transforms.ToTensor(),
					 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

cifar_train = dset.CIFAR10("./", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR10("./", train=False, transform=transform_test, target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=1, shuffle=False,num_workers=8,drop_last=False)

global count
count = 0

def imshow(img,i):
	global count
	npimg = img.numpy()

	gaussian_noise_0 = np.random.normal(0,args.gaussian_std,1296)
	gaussian_noise_1 = np.random.normal(0,args.gaussian_std,1296)
	gaussian_noise_2 = np.random.normal(0,args.gaussian_std,1296)

	npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std) + gaussian_noise_0.reshape(36,36)
	npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std) + gaussian_noise_1.reshape(36,36)
	npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std) + gaussian_noise_2.reshape(36,36)
	
	'''
	npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std)
	npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std)
	npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std)
	'''

	## add gaussian noise using numpy or scipy
	if args.test:
		foldername = '/home/yhbyun/dirtydataset/cifar10_gaussian_{}_blur_{}_test'.format(args.gaussian_std, args.blur_std)
		#foldername = '/home/yhbyun/dirtydataset/tmp'
		if not os.path.isdir(foldername):
			os.mkdir(foldername)
			assert os.path.isdir(foldername), 'Error: no checkpoint directory found!'
		#scipy.misc.toimage(np.transpose(npimg, (1,2,0))[2:34,2:34]).save(foldername+"/gaussian_{}_blur_{}_test_{i}.png".format(args.gaussian_std,args.blur_std,i=i))
		scipy.misc.toimage(np.transpose(npimg, (1,2,0))).save(foldername+"/cifar10_gaussian_{}_blur_{}_test_{i}.png".format(args.gaussian_std,args.blur_std,i=i))
		data_len = 10000

	if args.train:
		foldername = '/home/yhbyun/dirtydataset/cifar10_gaussian_{}_blur_{}_train'.format(args.gaussian_std, args.blur_std)
		if not os.path.isdir(foldername):
			os.mkdir(foldername)
			assert os.path.isdir(foldername), 'Error: no checkpoint directory found!'
		#scipy.misc.toimage(np.transpose(npimg, (1,2,0))).save(foldername+"/cifar10_gaussian_{}_blur_{}_train_{i}.png".format(args.gaussian_std,args.blur_std,i=i))
		scipy.misc.toimage(np.transpose(npimg, (1,2,0))).save(foldername+"/cifar10_gaussian_{}_blur_{}_train_{i}.png".format(args.gaussian_std,args.blur_std,i=i))
		data_len = 50000
	
	progress_bar(batch_idx, data_len);

if args.test:
	for batch_idx, (inputs, targets) in enumerate(test_loader):
	#if batch_idx < 10:
		'''global count
		if batch_idx == 446:
			imshow(torchvision.utils.make_grid(inputs),batch_idx)
		elif batch_idx == 448:
			exit()
		else:
			count += 1
		'''
		imshow(torchvision.utils.make_grid(inputs),batch_idx)

if args.train:
	for batch_idx, (inputs, targets) in enumerate(train_loader):
	#if batch_idx < 10:
		imshow(torchvision.utils.make_grid(inputs),batch_idx)

print('\n')
