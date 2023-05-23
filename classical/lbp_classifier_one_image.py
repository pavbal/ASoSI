import os

import skimage
from skimage.feature import local_binary_pattern
from sklearn import datasets
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import metrics
from methods import methods
from methods.methods import plot_two_with_map, compute_iou

data_matrix = np.load('./saved/dataset_rgb_1im.npy')
target_vect = np.ravel(np.load('./saved/mask_vect_rgb_1im.npy'))

print(data_matrix.shape)
print(target_vect.shape)
dim = 3

# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)

test_data = np.zeros((1024*1024, dim), dtype=int)
test_target = np.zeros((1024*1024, 1), dtype=int)

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
global_counter = 0
max_iter = 100

# bayes
gnb = sklearn.naive_bayes.GaussianNB(priors=None)
gnb.fit(data_matrix, target_vect)
# print(gnb.n_features_in_)

# SC
svc = svm.SVC(max_iter=max_iter)
svc.fit(data_matrix, target_vect)
# print(svc.n_features_in_)

im_i = skimage.io.imread("./LoveDA_Train_16/Rural/images_png/6.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Train_16/Rural/masks_png/6.png")

# plt.imshow(im_i)
# plt.show()


im_lbp_rgb = np.copy(im_i)
im_lbp_rgb[:, :, 0] = local_binary_pattern(im_i[:, :, 0], n_points, radius, METHOD)
im_lbp_rgb[:, :, 1] = local_binary_pattern(im_i[:, :, 1], n_points, radius, METHOD)
im_lbp_rgb[:, :, 2] = local_binary_pattern(im_i[:, :, 2], n_points, radius, METHOD)

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

# odtud je to přidáno později
dim = 3
number_images = 1
dataset_lbp_test = np.zeros((1024 * 1024 * number_images, dim), dtype=int)
mask_vect = np.zeros((1024*1024*number_images, 1), dtype=int)
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
#         im_lbp_rgb = im_i
#         im_lbp_rgb[:, :, 0] = local_binary_pattern(im_i[:, :, 0], n_points, radius, METHOD)
#         im_lbp_rgb[:, :, 1] = local_binary_pattern(im_i[:, :, 1], n_points, radius, METHOD)
#         im_lbp_rgb[:, :, 2] = local_binary_pattern(im_i[:, :, 2], n_points, radius, METHOD)

for x in range(np.size(im_m, 0)):
        for y in range(np.size(im_m, 1)):
            if im_m[x,y] != 0:
                dataset_lbp_test[global_counter, 0] = im_lbp_rgb[x, y, 0]
                dataset_lbp_test[global_counter, 1] = im_lbp_rgb[x, y, 1]
                dataset_lbp_test[global_counter, 2] = im_lbp_rgb[x, y, 2]
                mask_vect[global_counter, 0] = im_m[x, y]

                global_counter += 1

print(dataset_lbp_test)
print(dataset_lbp_test.shape)
# np.save('./saved/dataset_lbp_2', dataset_lbp_test)
# np.load('./saved/dataset_lbp.npy')
# dataset_lbp = dataset_lbp[~np.all(dataset_lbp == 0, axis=1)]
# print(dataset_lbp.size)
data_vect = dataset_lbp_test
target_vect = mask_vect
# np.save('./saved/data_vect_lbp_2', data_vect)
# np.save('./saved/target_vect_lbp_2', target_vect)

# y_pred = svc.predict(data_vect.reshape(-1, 1))
y_pred = gnb.predict(data_vect)#.reshape(-1, 1)
y_pred = svc.predict(data_vect)#.reshape(-1, 1)

target_vect = target_vect.reshape(-1)

print("target shape: ", np.shape(target_vect))
print("y_pred shape: ", np.shape(y_pred))
miou = metrics.miou(target_vect, y_pred)
acc = np.sum(y_pred==target_vect) / len(y_pred)
f1 = methods.f1_score(target_vect, y_pred)

print("average accuracy: ", acc)
print("mean IoU: ", miou)
print("average F1-score: ", f1)

global_counter = 0
#  zde končí přidaná sekce


for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x, y] != 0:
            test_data[global_counter, 0] = im_lbp_rgb[x, y, 0]
            test_data[global_counter, 1] = im_lbp_rgb[x, y, 1]
            test_data[global_counter, 2] = im_lbp_rgb[x, y, 2]
            if dim == 4:
                test_data[global_counter, 3] = lbp[x, y] * 2
            test_target[global_counter,] = im_m[x, y]
        global_counter += 1

# y_pred = gnb.predict(test_data).reshape(-1, 1) # bayes
y_pred = svc.predict(test_data).reshape(-1, 1)  # SVM


# print(y_pred.shape)
# print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
# print(test_target != y_pred)
print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
print("Percentage correct: ", (np.sum(test_target == y_pred)/(1024*1024)))
# print("IoU: ", compute_iou(y_pred, test_target))
# test_data_img = np.reshape(skimage.color.rgb2gray(test_data), (1024, 1024))
# test_data_img = np.reshape(test_data[:,1], (1024, 1024))
y_pred_img = np.reshape(y_pred, (1024, 1024))
# print("IoU: ", compute_iou(y_pred_img, im_m))
print("IoU_new: ", metrics.miou(im_m, y_pred_img))
print("F1: ", methods.f1_score(y_pred_img, im_m))

# # Vykresleni---------------------------------------------------------------------
#
# start_y = 300
# dest_y = start_y + 128
# start_x = 650
# dest_x = start_x + 128
#
# plot_two_with_map(im_m, y_pred_img, im_i=im_i,
#                   title="Predikce metody LBP (RGB) na tomtéž snímku - GNB",
#                   save_name="./figures/LBP_rgb_1img_SVM-5iter_1024_correct.png"
#                   # save_name="./figures/LBP_rgb_1img_GNB_1024_correct.png"
#                   )
#
# plot_two_with_map(im_m[start_x:dest_x, start_y:dest_y], y_pred_img[start_x:dest_x, start_y:dest_y], im_i=im_i[start_x:dest_x, start_y:dest_y],
#                   title="Predikce metody LBP (RGB) na tomtéž snímku - GNB, výřez",
#                   save_name="./figures/LBP_rgb_1img_SVM-5iter_128_correct.png"
#                   )



