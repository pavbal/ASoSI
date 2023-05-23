import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage


# Image loading
img =  io.imread('./LoveDA/Train/Rural/images_png/0.png')


# LBP calculation - original image
from skimage.feature import local_binary_pattern

radius = 1
n_points = 8 * radius
METHOD = 'uniform'

lbp = local_binary_pattern(skimage.color.rgb2gray(img), n_points, radius, METHOD)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')


ax2.axis('off')
ax2.imshow(lbp, cmap=plt.cm.gray)
ax2.set_title('Local Binary Pattern')
plt.show()


