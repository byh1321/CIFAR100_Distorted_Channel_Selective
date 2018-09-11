import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

img1 = cv2.imread('/home/yhbyun/dirtydataset/tmp/gaussian_0.0_blur_0.0_test_446.png', -1)
img2 = cv2.imread('/home/yhbyun/dirtydataset/tmp/gaussian_0.0_blur_0.5_test_446.png', -1)
img3 = cv2.imread('/home/yhbyun/dirtydataset/tmp/gaussian_0.0_blur_1.0_test_446.png', -1)
img4 = cv2.imread('/home/yhbyun/dirtydataset/tmp/gaussian_0.0_blur_1.5_test_446.png', -1)

color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img4],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale picture')
plt.show()

plt.figure(2)


while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()
