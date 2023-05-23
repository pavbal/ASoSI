import os

import skimage
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import local_binary_pattern
from sklearn import datasets
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt

import metrics
from methods import methods

data_vect_lbp = np.load('./saved/dataset_rgb_1im_gs.npy').reshape(-1, 1)
target_vect_lbp = np.load('./saved/mask_vect_rgb_1im_gs.npy').ravel()#reshape(-1, 1)

print(data_vect_lbp.shape)
print(target_vect_lbp.shape)

test_data = np.zeros((1024*1024, 1), dtype=int)
test_target = np.zeros((1024*1024, 1), dtype=int)

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
global_counter = 0
max_iter = 10

# GNB
gnb = sklearn.naive_bayes.GaussianNB()
gnb.fit(data_vect_lbp, target_vect_lbp)

# SVM
svc = svm.LinearSVC(max_iter=max_iter)
svc.fit(data_vect_lbp, target_vect_lbp)

im_i = skimage.io.imread("./LoveDA_Train_16/Rural/images_png/6.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Train_16/Rural/masks_png/6.png")

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

# odtud je to přidáno později
number_images = 1
dataset_lbp_test = np.zeros((1024 * 1024 * number_images, 2), dtype=int)
folder_dir_base = "./LoveDA_Test_16/"





# for folder_level_1 in sorted(os.listdir(folder_dir_base), key=len):
#     folder_dir_1 = folder_dir_base + folder_level_1 + "/"
#
#     folder_dir_2 = folder_dir_1 + "images_png" + "/"
#
#     for image in sorted(os.listdir(folder_dir_2), key=len):
#         file_name_image = "LoveDA_Test_16/" + folder_level_1 + "/images_png/" + image
#         file_name_mask = "LoveDA_Test_16/" + folder_level_1 + "/masks_png/" + image
#         im_i = skimage.io.imread(file_name_image)
#         im_i_g = skimage.color.rgb2gray(im_i)
#         im_m = skimage.io.imread(file_name_mask)
#
#         lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x,y] != 0:
            dataset_lbp_test[global_counter, 0] = lbp[x, y]
            dataset_lbp_test[global_counter, 1] = im_m[x, y]
            global_counter += 1
print(dataset_lbp_test)
print(dataset_lbp_test.shape)
# np.save('./saved/dataset_lbp_2', dataset_lbp_test)
# np.load('./saved/dataset_lbp.npy')
# dataset_lbp = dataset_lbp[~np.all(dataset_lbp == 0, axis=1)]
# print(dataset_lbp.size)
data_vect = dataset_lbp_test[:, 0]
target_vect = dataset_lbp_test[:, 1]
# np.save('./saved/data_vect_lbp_2', data_vect)
# np.save('./saved/target_vect_lbp_2', target_vect)

y_pred = svc.predict(data_vect.reshape(-1, 1))
y_pred2 = gnb.predict(data_vect.reshape(-1, 1))

print("target shape: ", np.shape(target_vect))
print("y_pred shape: ", np.shape(y_pred))
miou = metrics.miou(target_vect, y_pred)
acc = np.sum(y_pred==target_vect) / len(y_pred)
f1 = methods.f1_score(target_vect, y_pred)

print("average accuracy: ", acc)
print("mean IoU: ", miou)
print("average F1-score: ", f1)
print((y_pred))
print(np.sum(y_pred!=y_pred2))
print("")
print(np.sum(y_pred==7))
print(np.sum(y_pred==6))
print(np.sum(y_pred==5))
print(np.sum(y_pred==4))
print(np.sum(y_pred==3))
print(np.sum(y_pred==2))
print(np.sum(y_pred==1))




#  zde končí přidaná sekce

global_counter = 0
for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x, y] != 0:
            test_data[global_counter, ] = lbp[x, y]
            test_target[global_counter, ] = im_m[x, y]
        global_counter += 1

# y_pred = gnb.predict(test_data).reshape(-1, 1)
y_pred = svc.predict(test_data).reshape(-1, 1)
print(np.shape(y_pred))
# print(y_pred.shape)
# print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
# print(test_target != y_pred)

test_data_img = np.reshape(test_data, (1024, 1024))
y_pred_img = np.reshape(y_pred, (1024, 1024))

result = svc.score(y_pred, test_target)
print(result)
