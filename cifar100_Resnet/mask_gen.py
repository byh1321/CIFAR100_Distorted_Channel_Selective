import torch

mask=[]
mask_conv1 = torch.randn(64,3,3,3).cuda()/10
mask_1_1 = torch.randn(64,64,3,3).cuda()/10
mask_1_2 = torch.randn(64,64,3,3).cuda()/10
mask_1_3 = torch.randn(64,64,3,3).cuda()/10
mask_1_4 = torch.randn(64,64,3,3).cuda()/10

mask_2_1 = torch.randn(128,64,3,3).cuda()/10
mask_2_2 = torch.randn(128,128,3,3).cuda()/10
mask_2_3 = torch.randn(128,128,3,3).cuda()/10
mask_2_4 = torch.randn(128,128,3,3).cuda()/10

mask_3_1 = torch.randn(256,128,3,3).cuda()/10
mask_3_2 = torch.randn(256,256,3,3).cuda()/10
mask_3_3 = torch.randn(256,256,3,3).cuda()/10
mask_3_4 = torch.randn(256,256,3,3).cuda()/10

mask_4_1 = torch.randn(512,256,3,3).cuda()/10
mask_4_2 = torch.randn(512,512,3,3).cuda()/10
mask_4_3 = torch.randn(512,512,3,3).cuda()/10
mask_4_4 = torch.randn(512,512,3,3).cuda()/10

mask_1_down = torch.randn(128,64,1,1).cuda()/10
mask_2_down = torch.randn(256,128,1,1).cuda()/10
mask_3_down = torch.randn(512,256,1,1).cuda()/10

mask_linear = torch.randn(100,512).cuda()/10

mask.append(mask_conv1)
mask.append(mask_1_1)
mask.append(mask_1_2)
mask.append(mask_1_3)
mask.append(mask_1_4)
mask.append(mask_2_1)
mask.append(mask_2_2)
mask.append(mask_2_3)
mask.append(mask_2_4)
mask.append(mask_3_1)
mask.append(mask_3_2)
mask.append(mask_3_3)
mask.append(mask_3_4)
mask.append(mask_4_1)
mask.append(mask_4_2)
mask.append(mask_4_3)
mask.append(mask_4_4)
mask.append(mask_1_down)
mask.append(mask_2_down)
mask.append(mask_3_down)
mask.append(mask_linear)

torch.save(mask, 'mask_rand.dat')
