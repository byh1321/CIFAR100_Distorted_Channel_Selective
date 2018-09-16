import torch
import torch.nn as nn
from torch.autograd import Variable

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
			nn.Linear(512, 100)
		)

	def forward(self,x):
		fixed = 0
		if fixed:
			x = quant(x)
			x = roundmax(x)

		out = self.conv1(x)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu2(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu4(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_conv6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu6(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		residual = self.layer2_downsample(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu2(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu4(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu6(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv7(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_conv8(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu8(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		residual = self.layer3_downsample(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu2(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu4(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu6(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv7(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv8(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu8(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv9(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv10(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu10(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv11(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_conv12(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu12(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		residual = self.layer4_downsample(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu2(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu4(out)

		residual = out

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_conv6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu6(out)

		out = self.avgpool(out)

		out = out.view(out.size(0), -1)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.linear(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		return out

def roundmax(input):
	maximum = 2**args.iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	return input	

def quant(input):
	input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return input

