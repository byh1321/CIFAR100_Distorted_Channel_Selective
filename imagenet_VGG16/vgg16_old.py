import torch
import torch.nn as nn
from torch.autograd import Variable

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
