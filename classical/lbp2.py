import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.color import rgba2rgb, rgb2gray
import skimage.segmentation
import scipy
# from scipy.misc import imread
import matplotlib.image as img
import skimage.io
import os
import fnmatch
from skimage.feature import local_binary_pattern


#parametry lbp
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
number_images = 16

folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA_Train_16/"
folder_dir_base = "./LoveDA_Train_16/"
dataset_lbp = np.zeros((1024*1024*number_images, 2), dtype=int)


global_counter = 0


for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"

    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in os.listdir(folder_dir_2):
        file_name_image = "LoveDA_Train_16/" + folder_level_1 + "/images_png/" + image
        file_name_mask = "LoveDA_Train_16/" + folder_level_1 + "/masks_png/" + image
        im_i = skimage.io.imread(file_name_image)
        im_i_g = skimage.color.rgb2gray(im_i)
        im_m = skimage.io.imread(file_name_mask)

        lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

        for x in range(np.size(im_m, 0)):
            for y in range(np.size(im_m, 1)):
                if im_m[x,y] != 0:
                    dataset_lbp[global_counter, 0] = lbp[x,y]
                    dataset_lbp[global_counter, 1] = im_m[x,y]
                    global_counter += 1
    print(dataset_lbp)
    print(dataset_lbp.shape)
    np.save('./saved/dataset_lbp_2', dataset_lbp)
    np.load('./saved/dataset_lbp.npy')
    # dataset_lbp = dataset_lbp[~np.all(dataset_lbp == 0, axis=1)]
    # print(dataset_lbp.size)
    data_vect = dataset_lbp[:, 0]
    target_vect = dataset_lbp[:, 1]
    np.save('./saved/data_vect_lbp_2', data_vect)
    np.save('./saved/target_vect_lbp_2', target_vect)






