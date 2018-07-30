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

from utils import progress_bar

import os

parser = argparse.ArgumentParser(description='PyTorch ImageNet2012 VGG16 Pre-trained Quantization')
parser.add_argument('--pprec', type=int, default=20, metavar='N', help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N', help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N', help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N', help='fixed=0 - floating point arithmetic')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

valdir = os.path.join("/home/mhha/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

net = models.vgg16(pretrained=True)

net2 = VGG16()

print("saving & quantizing CONV1 weights")
net2.conv1[0].weight = net.features[0].weight
net2.conv1[0].bias = net.features[0].bias

if args.fixed:
    net2.conv1[0].weight.data = torch.round(net2.conv1[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv1[0].bias.data = torch.round(net2.conv1[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV2 weights")
net2.conv2[0].weight = net.features[2].weight
net2.conv2[0].bias = net.features[2].bias

if args.fixed:
    net2.conv2[0].weight.data = torch.round(net2.conv2[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv2[0].bias.data = torch.round(net2.conv2[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))
    
print("saving & quantizing CONV3 weights")
net2.conv3[0].weight = net.features[5].weight
net2.conv3[0].bias = net.features[5].bias

if args.fixed:
    net2.conv3[0].weight.data = torch.round(net2.conv3[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv3[0].bias.data = torch.round(net2.conv3[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV4 weights")
net2.conv4[0].weight = net.features[7].weight
net2.conv4[0].bias = net.features[7].bias

if args.fixed:
    net2.conv4[0].weight.data = torch.round(net2.conv4[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv4[0].bias.data = torch.round(net2.conv4[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV5 weights")
net2.conv5[0].weight = net.features[10].weight
net2.conv5[0].bias = net.features[10].bias

if args.fixed:
    net2.conv5[0].weight.data = torch.round(net2.conv5[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv5[0].bias.data = torch.round(net2.conv5[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV6 weights")
net2.conv6[0].weight = net.features[12].weight
net2.conv6[0].bias = net.features[12].bias

if args.fixed:
    net2.conv6[0].weight.data = torch.round(net2.conv6[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv6[0].bias.data = torch.round(net2.conv6[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV7 weights")
net2.conv7[0].weight = net.features[14].weight
net2.conv7[0].bias = net.features[14].bias

if args.fixed:
    net2.conv7[0].weight.data = torch.round(net2.conv7[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv7[0].bias.data = torch.round(net2.conv7[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV8 weights")
net2.conv8[0].weight = net.features[17].weight
net2.conv8[0].bias = net.features[17].bias

if args.fixed:
    net2.conv8[0].weight.data = torch.round(net2.conv8[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv8[0].bias.data = torch.round(net2.conv8[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV9 weights")
net2.conv9[0].weight = net.features[19].weight
net2.conv9[0].bias = net.features[19].bias

if args.fixed:
    net2.conv9[0].weight.data = torch.round(net2.conv9[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv9[0].bias.data = torch.round(net2.conv9[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV10 weights")
net2.conv10[0].weight = net.features[21].weight
net2.conv10[0].bias = net.features[21].bias

if args.fixed:
    net2.conv10[0].weight.data = torch.round(net2.conv10[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv10[0].bias.data = torch.round(net2.conv10[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV11 weights")
net2.conv11[0].weight = net.features[24].weight
net2.conv11[0].bias = net.features[24].bias

if args.fixed:
    net2.conv11[0].weight.data = torch.round(net2.conv11[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv11[0].bias.data = torch.round(net2.conv11[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV12 weights")
net2.conv12[0].weight = net.features[26].weight
net2.conv12[0].bias = net.features[26].bias

if args.fixed:
    net2.conv12[0].weight.data = torch.round(net2.conv12[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv12[0].bias.data = torch.round(net2.conv12[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing CONV13 weights")
net2.conv13[0].weight = net.features[28].weight
net2.conv13[0].bias = net.features[28].bias

if args.fixed:
    net2.conv13[0].weight.data = torch.round(net2.conv13[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.conv13[0].bias.data = torch.round(net2.conv13[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing FC1 weights")
net2.linear1[0].weight = net.classifier[0].weight
net2.linear1[0].bias = net.classifier[0].bias

if args.fixed:
    net2.linear1[0].weight.data = torch.round(net2.linear1[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.linear1[0].bias.data = torch.round(net2.linear1[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing FC2 weights")
net2.linear2[0].weight = net.classifier[3].weight
net2.linear2[0].bias = net.classifier[3].bias

if args.fixed:
    net2.linear2[0].weight.data = torch.round(net2.linear2[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.linear2[0].bias.data = torch.round(net2.linear2[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

print("saving & quantizing FC3 weights")
net2.linear3[0].weight = net.classifier[6].weight
net2.linear3[0].bias = net.classifier[6].bias

if args.fixed:
    net2.linear3[0].weight.data = torch.round(net2.linear3[0].weight.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if args.fixed:
    net2.linear3[0].bias.data = torch.round(net2.linear3[0].bias.data / (2 ** -(args.pprec))) * (2 ** -(args.pprec))

if use_cuda:
    print(torch.cuda.device_count())
    net2.cuda()
    net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net2.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net2(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.data
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
            'net2': net2.module if use_cuda else net2,
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_vgg16_layer_by_layer_v2.t7')
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

print("Layer by layer VGG16 Quant Test, pprec: %d, iwidth: %d, aprec: %d \n" % (args.pprec, args.iwidth, args.aprec))
test()