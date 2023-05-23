import skimage
import numpy as np

im_i = skimage.io.imread("./Result/4191.png")
print(np.sum(im_i==5))