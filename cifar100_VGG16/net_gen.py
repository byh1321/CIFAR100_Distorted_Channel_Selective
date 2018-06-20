import torch

layer=[]

layer_conv0 = torch.zeros(64,3,3,3).cuda()
layer_conv1 = torch.zeros(64,64,3,3).cuda()

layer_conv2 = torch.zeros(128,64,3,3).cuda()
layer_conv3 = torch.zeros(128,128,3,3).cuda()

layer_conv4 = torch.zeros(256,128,3,3).cuda()
layer_conv5 = torch.zeros(256,256,3,3).cuda()
layer_conv6 = torch.zeros(256,256,3,3).cuda()

layer_conv7 = torch.zeros(512,256,3,3).cuda()
layer_conv8 = torch.zeros(512,512,3,3).cuda()
layer_conv9 = torch.zeros(512,512,3,3).cuda()
layer_conv10 = torch.zeros(512,512,3,3).cuda()
layer_conv11 = torch.zeros(512,512,3,3).cuda()
layer_conv12 = torch.zeros(512,512,3,3).cuda()
layer_fc0 = torch.zeros(512,512).cuda()
layer_fc1 = torch.zeros(512,512).cuda()

layer_fc2 = torch.zeros(100,512).cuda()

layer.append(layer_conv0)
layer.append(layer_conv1)
layer.append(layer_conv2)
layer.append(layer_conv3)
layer.append(layer_conv4)
layer.append(layer_conv5)
layer.append(layer_conv6)
layer.append(layer_conv7)
layer.append(layer_conv8)
layer.append(layer_conv9)
layer.append(layer_conv10)
layer.append(layer_conv11)
layer.append(layer_conv12)
layer.append(layer_fc0)
layer.append(layer_fc1)
layer.append(layer_fc2)

torch.save(layer, 'layer_null.dat')
