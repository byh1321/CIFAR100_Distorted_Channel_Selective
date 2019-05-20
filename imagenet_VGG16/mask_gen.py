import torch

mask=[]
mask_conv0 = torch.zeros(64,3,3,3).cuda()/10
mask_batchnorm0 = torch.zeros(64)
mask_conv1 = torch.zeros(64,64,3,3).cuda()/10
mask_batchnorm1 = torch.zeros(64)

mask_conv2 = torch.zeros(128,64,3,3).cuda()/10
mask_batchnorm2 = torch.zeros(128)
mask_conv3 = torch.zeros(128,128,3,3).cuda()/10
mask_batchnorm3 = torch.zeros(128)

mask_conv4 = torch.zeros(256,128,3,3).cuda()/10
mask_batchnorm4 = torch.zeros(256)
mask_conv5 = torch.zeros(256,256,3,3).cuda()/10
mask_batchnorm5 = torch.zeros(256)
mask_conv6 = torch.zeros(256,256,3,3).cuda()/10
mask_batchnorm6 = torch.zeros(256)

mask_conv7 = torch.zeros(512,256,3,3).cuda()/10
mask_batchnorm7 = torch.zeros(512)
mask_conv8 = torch.zeros(512,512,3,3).cuda()/10
mask_batchnorm8 = torch.zeros(512)
mask_conv9 = torch.zeros(512,512,3,3).cuda()/10
mask_batchnorm9 = torch.zeros(512)
mask_conv10 = torch.zeros(512,512,3,3).cuda()/10
mask_batchnorm10 = torch.zeros(512)
mask_conv11 = torch.zeros(512,512,3,3).cuda()/10
mask_batchnorm11 = torch.zeros(512)
mask_conv12 = torch.zeros(512,512,3,3).cuda()/10
mask_batchnorm12 = torch.zeros(512)
mask_fc0 = torch.zeros(4096,25088).cuda()/10
mask_fc1 = torch.zeros(4096,4096).cuda()/10

mask_fc2 = torch.zeros(1000,4096).cuda()/10

mask.append(mask_conv0)
mask.append(mask_batchnorm0)
mask.append(mask_conv1)
mask.append(mask_batchnorm1)
mask.append(mask_conv2)
mask.append(mask_batchnorm2)
mask.append(mask_conv3)
mask.append(mask_batchnorm3)
mask.append(mask_conv4)
mask.append(mask_batchnorm4)
mask.append(mask_conv5)
mask.append(mask_batchnorm5)
mask.append(mask_conv6)
mask.append(mask_batchnorm6)
mask.append(mask_conv7)
mask.append(mask_batchnorm7)
mask.append(mask_conv8)
mask.append(mask_batchnorm8)
mask.append(mask_conv9)
mask.append(mask_batchnorm9)
mask.append(mask_conv10)
mask.append(mask_batchnorm10)
mask.append(mask_conv11)
mask.append(mask_batchnorm11)
mask.append(mask_conv12)
mask.append(mask_batchnorm12)
mask.append(mask_fc0)
mask.append(mask_fc1)
mask.append(mask_fc2)

torch.save(mask, 'mask_with_batch.dat')
