import torch
import torch.nn as nn
from torch.autograd import Variable

class ResNet34(nn.Module):
	def __init__(self):
		super(ResNet34,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
		)
		self.layer1_basic1 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_basic2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)
		self.layer1_relu1 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer1_basic3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_basic4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)
		self.layer1_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer1_basic5 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.layer1_basic6 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)
		self.layer1_relu3 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer2_basic1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_downsample = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_basic2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_relu1 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer2_basic3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_basic4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer2_basic5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_basic6 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_relu3 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer2_basic7 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.layer2_basic8 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
		)
		self.layer2_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)

		self.layer3_basic1 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_downsample = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_basic2 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu1 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer3_basic3 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_basic4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer3_basic5 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_basic6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu3 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer3_basic7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_basic8 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu4 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer3_basic9 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_basic10 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu5 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer3_basic11 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.layer3_basic12 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
		)
		self.layer3_relu6 = nn.Sequential(
			nn.ReLU(inplace=True),
		)


		self.layer4_basic1 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_downsample = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(512),
		)
		self.layer4_basic2 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)
		self.layer4_relu1 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer4_basic3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_basic4 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)
		self.layer4_relu2 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.layer4_basic5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.layer4_basic6 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
		)
		self.layer4_relu3 = nn.Sequential(
			nn.ReLU(inplace=True),
		)
		self.avgpool = nn.Sequential(
			nn.AvgPool2d(2, stride=1, padding=0)
		)
		self.linear = nn.Sequential(
			nn.Linear(512, 100)
		)
		self._initialize_weights()

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

		out = self.layer1_basic1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu1(out)
		residual = out

		out = self.layer1_basic3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu2(out)

		residual = out

		out = self.layer1_basic5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu3(out)

		residual = self.layer2_downsample(out)

		out = self.layer2_basic1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)
		
		out += residual
		out = self.layer2_relu1(out)

		residual = out

		out = self.layer2_basic3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu2(out)

		residual = out

		out = self.layer2_basic5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu3(out)

		residual = out

		out = self.layer2_basic7(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic8(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu4(out)

		residual = self.layer3_downsample(out)

		out = self.layer3_basic1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu1(out)

		residual = out

		out = self.layer3_basic3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu2(out)

		residual = out

		out = self.layer3_basic5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu3(out)

		residual = out

		out = self.layer3_basic7(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic8(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu4(out)

		residual = out

		out = self.layer3_basic9(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic10(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu5(out)
		
		residual = out

		out = self.layer3_basic11(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic12(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu6(out)

		residual = self.layer4_downsample(out)

		out = self.layer4_basic1(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic2(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu1(out)
		residual = out

		out = self.layer4_basic3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic4(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu2(out)
		residual = out

		out = self.layer4_basic5(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic6(out)

		if fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu3(out)

		if fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.avgpool(out)
		#print(out.size())
		out = out.view(out.size(0), -1)
		#print(out.size())

		out = self.linear(out)

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

