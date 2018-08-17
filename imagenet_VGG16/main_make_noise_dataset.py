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

import time
import os
import numpy as np
import argparse

from PIL import Image
import glob
#from sklearn.preprocessing import normalize

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gaussian_std', default=0, type=float, help='std of gaussian noise')
parser.add_argument('--blur_std', default=0, type=float, help='std of blur noise')

args = parser.parse_args()

f = open('foldernames.txt','r')
clean = f.readlines()
f.close()
f = open('dirty_foldernames.txt','r')
dirty = f.readlines()
f.close()

def imshow(img,folder_idx, img_count):
	npimg = np.array(img, dtype=float)
	npimg[:,:,0] = (npimg[:,:,0] - np.mean(npimg[:,:,0]))/2/np.std(npimg[:,:,0])+0.5
	npimg[:,:,1] = (npimg[:,:,1] - np.mean(npimg[:,:,1]))/2/np.std(npimg[:,:,1])+0.5
	npimg[:,:,2] = (npimg[:,:,2] - np.mean(npimg[:,:,2]))/2/np.std(npimg[:,:,2])+0.5

	npimg_size = npimg.size
	npimg_shape = npimg.shape

	if args.blur_std == 0:

		'''
		try:
			gaussian_noise_0 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[2])
		except:
			print(img_count)
			print('npimg_size : ' + str(npimg_size))
			print('npimg_shape : ' + str(npimg_shape))
			exit()
		'''
		'''
		gaussian_noise_0 = np.random.normal(0,args.gaussian_std * np.std(npimg[:,:,0]),npimg_shape[1]*npimg_shape[0])
		gaussian_noise_1 = np.random.normal(0,args.gaussian_std * np.std(npimg[:,:,1]),npimg_shape[1]*npimg_shape[0])
		gaussian_noise_2 = np.random.normal(0,args.gaussian_std * np.std(npimg[:,:,2]),npimg_shape[1]*npimg_shape[0])
		npimg[:,:,0] = npimg[:,:,0] + gaussian_noise_0.reshape(npimg_shape[0],npimg_shape[1])
		npimg[:,:,1] = npimg[:,:,1] + gaussian_noise_1.reshape(npimg_shape[0],npimg_shape[1])
		npimg[:,:,2] = npimg[:,:,2] + gaussian_noise_2.reshape(npimg_shape[0],npimg_shape[1])
		'''
		gaussian_noise_0 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[0])
		gaussian_noise_1 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[0])
		gaussian_noise_2 = np.random.normal(0,args.gaussian_std,npimg_shape[1]*npimg_shape[0])
		npimg[:,:,0] = npimg[:,:,0] + gaussian_noise_0.reshape(npimg_shape[0],npimg_shape[1])
		npimg[:,:,1] = npimg[:,:,1] + gaussian_noise_1.reshape(npimg_shape[0],npimg_shape[1])
		npimg[:,:,2] = npimg[:,:,2] + gaussian_noise_2.reshape(npimg_shape[0],npimg_shape[1])
	elif args.gaussian_std == 0:
		npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std * np.std(npimg[0]))
		npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std * np.std(npimg[1]))
		npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std * np.std(npimg[2]))
	else:	
		gaussian_noise_0 = np.random.normal(0,args.gaussian_std * np.std(npimg[0]),npimg_shape[1]*npimg_shape[2])
		gaussian_noise_1 = np.random.normal(0,args.gaussian_std * np.std(npimg[1]),npimg_shape[1]*npimg_shape[2])
		gaussian_noise_2 = np.random.normal(0,args.gaussian_std * np.std(npimg[2]),npimg_shape[1]*npimg_shape[2])

		npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std) + gaussian_noise_0.reshape(npimg_shape[1],npimg_shape[2])
		npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std) + gaussian_noise_1.reshape(npimg_shape[1],npimg_shape[2])
		npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std) + gaussian_noise_2.reshape(npimg_shape[1],npimg_shape[2])
		
		npimg[0] = scipy.ndimage.filters.gaussian_filter(npimg[0], args.blur_std)
		npimg[1] = scipy.ndimage.filters.gaussian_filter(npimg[1], args.blur_std)
		npimg[2] = scipy.ndimage.filters.gaussian_filter(npimg[2], args.blur_std)
	
	## add gaussian noise using numpy or scipy
	str_end = len(dirty[folder_idx])-1
	foldername = dirty[folder_idx][0:str_end]
	if not os.path.isdir(foldername):
		os.mkdir(foldername)
		assert os.path.isdir(foldername), 'Error: no checkpoint directory found!'
	scipy.misc.toimage(npimg).save(foldername+"/imagenet_gaussian_{}_blur_{}_train_{i}.png".format(args.gaussian_std,args.blur_std,i=img_count))
	
f = open('dirty_image_failed.log','a+')
print(time.ctime())
print(time.ctime(), file=f)
for folder_idx in range(0,1000):
	
	image_loader = []
	
	str_end = len(clean[folder_idx])-1
	foldername = clean[folder_idx][0:str_end]

	for filename in glob.glob(foldername+'/*.JPEG'): #assuming gif
		im=Image.open(filename)
		image_loader.append(im)
	
	img_count = 0
	for img in image_loader:
		try:
			imshow(img, folder_idx, img_count)
		except:
			print('{}th folder, {}th img skipped'.format(folder_idx, img_count))
			print('{}th folder, {}th img skipped'.format(folder_idx, img_count), file = f)
		img_count += 1
		'''
		if img_count == 1:
			exit()
		'''
	progress_bar(folder_idx, 1000);
f.close()
print('\n')
