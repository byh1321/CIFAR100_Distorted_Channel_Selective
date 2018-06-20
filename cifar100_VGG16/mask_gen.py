import torch

mask=[]

mask_conv0 = torch.randn(64,3,3,3).cuda()
mask_conv1 = torch.randn(64,64,3,3).cuda()

mask_conv2 = torch.randn(128,64,3,3).cuda()
mask_conv3 = torch.randn(128,128,3,3).cuda()

mask_conv4 = torch.randn(256,128,3,3).cuda()
mask_conv5 = torch.randn(256,256,3,3).cuda()
mask_conv6 = torch.randn(256,256,3,3).cuda()

mask_conv7 = torch.randn(512,256,3,3).cuda()
mask_conv8 = torch.randn(512,512,3,3).cuda()
mask_conv9 = torch.randn(512,512,3,3).cuda()
mask_conv10 = torch.randn(512,512,3,3).cuda()
mask_conv11 = torch.randn(512,512,3,3).cuda()
mask_conv12 = torch.randn(512,512,3,3).cuda()
mask_fc0 = torch.randn(512,512).cuda()
mask_fc1 = torch.randn(512,512).cuda()

mask_fc2 = torch.randn(100,512).cuda()

mask.append(mask_conv0)
mask.append(mask_conv1)
mask.append(mask_conv2)
mask.append(mask_conv3)
mask.append(mask_conv4)
mask.append(mask_conv5)
mask.append(mask_conv6)
mask.append(mask_conv7)
mask.append(mask_conv8)
mask.append(mask_conv9)
mask.append(mask_conv10)
mask.append(mask_conv11)
mask.append(mask_conv12)
mask.append(mask_fc0)
mask.append(mask_fc1)
mask.append(mask_fc2)

torch.save(mask, 'mask_rand.dat')
