import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
		)
		self.layer1_basic1 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer1_basic3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer2_basic1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_downsample = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_basic2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer2_basic3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_basic4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic1 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_downsample = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_basic2 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic3 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_basic4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer4_basic1 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_downsample = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer4_basic2 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer4_basic3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_basic4 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.linear = nn.Sequential(
			nn.Linear(512, 100)
		)
		self._initialize_weights()

	def forward(self,x):
		fixed = 0
		'''
		if fixed:
			x = quant(x)
			x = roundmax(x)
		'''
		out = x.clone()
		out = self.conv1(out)

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
		out = F.avg_pool2d(out, 2)
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

