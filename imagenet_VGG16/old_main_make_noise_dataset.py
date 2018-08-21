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
import torchvision.datasets as datasets

import scipy.misc
from scipy import ndimage

import os
import numpy as np
import argparse

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gaussian_std', default=0, type=float, help='std of gaussian noise')
parser.add_argument('--blur_std', default=0, type=float, help='std of blur noise')
parser.add_argument('--val', default=0, type=float, help='make test dataset if test=1')
parser.add_argument('--train', default=0, type=float, help='make train dataset if train=1')

args = parser.parse_args()

transform_train = transforms.Compose([transforms.ToTensor(),
					  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
transform_test = transforms.Compose([transforms.ToTensor(),
					 transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

#cifar_train = dset.CIFAR100("./", train=True, transform=transform_train, target_transform=None, download=True)
#cifar_test = dset.CIFAR100("./", train=False, transform=transform_test, target_transform=None, download=True)

traindir = os.path.join('/home/yhbyun/Imagenet2012/','train')

train_dataset = datasets.ImageFolder(
	traindir,
	transforms.Compose([
		transforms.ToTensor(),
	]))

valdir = os.path.join('/home/mhha/','val')
val_dataset = datasets.ImageFolder(
	valdir,
	transforms.Compose([
	transforms.ToTensor(),
	]))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=False,num_workers=2,drop_last=False)
test_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False,num_workers=8,drop_last=False)
#test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=1, shuffle=False,num_workers=2,drop_last=False)

def imshow(img,i,size):
	npimg = img.numpy()
	#print(npimg.shape)
	npimg_size = npimg.size
	npimg_shape = npimg.shape
	gaussian_noise_0 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[2])
	gaussian_noise_1 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[2])
	gaussian_noise_2 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[2])
	
	'''	
	npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std) + gaussian_noise_0.reshape(npimg_shape[1],npimg_shape[2])
	npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std) + gaussian_noise_1.reshape(npimg_shape[1],npimg_shape[2])
	npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std) + gaussian_noise_2.reshape(npimg_shape[1],npimg_shape[2])
	
	npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std)
	npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std)
	npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std)
	'''
	if not(args.gaussian_std == 0):
		npimg[0] = npimg[0] + gaussian_noise_0.reshape(npimg_shape[1],npimg_shape[2])
		npimg[1] = npimg[1] + gaussian_noise_1.reshape(npimg_shape[1],npimg_shape[2])
		npimg[2] = npimg[2] + gaussian_noise_2.reshape(npimg_shape[1],npimg_shape[2])
	

	## add gaussian noise using numpy or scipy
	if args.val:
		foldername = '/data/imagenet_dirty/imagenet_gaussian_{}_blur_{}_val'.format(args.gaussian_std, args.blur_std)
		if not os.path.isdir(foldername):
			os.mkdir(foldername)
			assert os.path.isdir(foldername), 'Error: no checkpoint directory found!'
		scipy.misc.toimage(np.transpose(npimg, (1,2,0))).save(foldername+"/imagenet_gaussian_{}_blur_{}_val_{i}.png".format(args.gaussian_std,args.blur_std,i=i))

	if args.train:
		foldername = '/home/yhbyun/Imagenet2012/dirty/'.format(args.gaussian_std, args.blur_std)
		if not os.path.isdir(foldername):
			os.mkdir(foldername)
			assert os.path.isdir(foldername), 'Error: no checkpoint directory found!'
		scipy.misc.toimage(np.transpose(npimg, (1,2,0))).save(foldername+"/imagenet_gaussian_{}_train_{}_train_{i}.png".format(args.gaussian_std,args.blur_std,i=i))
	
	progress_bar(batch_idx,size)

if args.val:
	for batch_idx, (inputs, targets) in enumerate(test_loader):
	#if batch_idx < 10:
		imshow(torchvision.utils.make_grid(inputs),batch_idx, len(test_loader))

if args.train:
	for batch_idx, (inputs, targets) in enumerate(train_loader):
	#if batch_idx < 10:
		imshow(torchvision.utils.make_grid(inputs),batch_idx, len(train_loader))

print('\n')
