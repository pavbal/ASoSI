import os

import skimage
from skimage.feature import local_binary_pattern
#from sklearn import datasets
import numpy as np

# import sklearn.model_selection
# from sklearn import svm
# import sklearn.naive_bayes
# import matplotlib.pyplot as plt


radius = 1
n_points = 8 * radius
METHOD = 'uniform'
number_images = 1

dim = 3


im_i = skimage.io.imread("./LoveDA_Test_16/Rural/images_png/17.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Test_16/Rural/masks_png/17.png")

print(im_i.shape)

im_i_red = im_i[:,:,0]
im_i_green = im_i[:,:,1]
im_i_blue = im_i[:,:,2]

lbp_red = local_binary_pattern(im_i_red, n_points, radius, METHOD)
lbp_green = local_binary_pattern(im_i_green, n_points, radius, METHOD)
lbp_blue = local_binary_pattern(im_i_blue, n_points, radius, METHOD)

im_lbp_rgb = im_i
im_lbp_rgb[:,:,0] = local_binary_pattern(im_i[:,:,0], n_points, radius, METHOD)
im_lbp_rgb[:,:,1] = local_binary_pattern(im_i[:,:,1], n_points, radius, METHOD)
im_lbp_rgb[:,:,2] = local_binary_pattern(im_i[:,:,2], n_points, radius, METHOD)

data_rgb = np.squeeze(np.reshape(im_lbp_rgb, (1024*1024, -1, 3)))

print(data_rgb.shape)


folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA_Train_16/"
folder_dir_base = "./LoveDA_Train_16/"
dataset_lbp = np.zeros((1024*1024*number_images, dim), dtype=int)
mask_vect = np.zeros((1024*1024*number_images, 1), dtype=int)
global_counter = 0
print(dataset_lbp.shape)

file_name_image = "./LoveDA_Train_16/Rural/images_png/6.png"
file_name_mask = "./LoveDA_Train_16/Rural/masks_png/6.png"
im_i = skimage.io.imread(file_name_image)
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread(file_name_mask)

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)
im_lbp_rgb = im_i
im_lbp_rgb[:, :, 0] = local_binary_pattern(im_i[:, :, 0], n_points, radius, METHOD)
im_lbp_rgb[:, :, 1] = local_binary_pattern(im_i[:, :, 1], n_points, radius, METHOD)
im_lbp_rgb[:, :, 2] = local_binary_pattern(im_i[:, :, 2], n_points, radius, METHOD)


for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x,y] != 0:
            dataset_lbp[global_counter, 0] = im_lbp_rgb[x,y,0]
            dataset_lbp[global_counter, 1] = im_lbp_rgb[x,y,1]
            dataset_lbp[global_counter, 2] = im_lbp_rgb[x,y,2]
            if dim == 4:
                dataset_lbp[global_counter, 2] = lbp[x, y] * 2
            mask_vect[global_counter, 0] = im_m[x,y]
            global_counter += 1
print(dataset_lbp.shape)
np.save('./saved/dataset_rgb_1im', dataset_lbp)
np.save('./saved/mask_vect_rgb_1im', mask_vect)
