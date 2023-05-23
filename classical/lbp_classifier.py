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

data_vect_lbp = np.load('./saved/data_vect_lbp_2.npy').reshape(-1, 1)
target_vect_lbp = np.load('./saved/target_vect_lbp_2.npy')

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

im_i = skimage.io.imread("./LoveDA_Test_16/Rural/images_png/23.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Test_16/Rural/masks_png/23.png")

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

# odtud je to přidáno později
number_images = 16
dataset_lbp_test = np.zeros((1024 * 1024 * number_images, 2), dtype=int)
folder_dir_base = "./LoveDA_Test_16/"
for folder_level_1 in sorted(os.listdir(folder_dir_base), key=len):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"

    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in sorted(os.listdir(folder_dir_2), key=len):
        file_name_image = "LoveDA_Test_16/" + folder_level_1 + "/images_png/" + image
        file_name_mask = "LoveDA_Test_16/" + folder_level_1 + "/masks_png/" + image
        im_i = skimage.io.imread(file_name_image)
        im_i_g = skimage.color.rgb2gray(im_i)
        im_m = skimage.io.imread(file_name_mask)

        lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

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

# y_pred = svc.predict(data_vect.reshape(-1, 1))
y_pred = gnb.predict(data_vect.reshape(-1, 1))

print("target shape: ", np.shape(target_vect))
print("y_pred shape: ", np.shape(y_pred))
miou = metrics.miou(target_vect, y_pred)
acc = np.sum(y_pred==target_vect) / len(y_pred)
f1 = methods.f1_score(target_vect, y_pred)

print("average accuracy: ", acc)
print("mean IoU: ", miou)
print("average F1-score: ", f1)

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


from methods.methods import smooth_mask, plot_two_with_map

# num_classes = 8
# colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
#
# myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
# class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
#
# # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# fig.suptitle('Výsledek metody LBP - 1D', fontsize=14, )
#
#
# ax1.axis('off')
# ax1.imshow(smooth_mask(im_m)[0:256, 0:256], cmap=myCmap, vmin=0, vmax=num_classes - 1)
# ax1.set_title('Originální maska snímku')
#
# ax2.axis('off')
# im = ax2.imshow(y_pred_img[0:256, 0:256], cmap=myCmap, vmin=0, vmax=num_classes - 1)
# ax2.imshow(y_pred_img[0:256, 0:256], cmap=myCmap, vmin=0, vmax=num_classes - 1)
# ax2.set_title('Predikovaná maska')
#
#
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('right', size='5%', pad=0.1)
# cb = fig.colorbar(im, cax=cax, ticks=class_labels)
#
# # navíc
# ticks = (np.arange(len(descriptions)) + 0.5)/1.14
# cb.set_ticks(ticks)
# cb.set_ticklabels(descriptions)
# cb.ax.tick_params(labelsize=12)
#
# cb.ax.set_title('Třídy', fontsize=12)
# # plt.savefig('images/figure.svg', format='svg', bbox_inches='tight', dpi=300)
# # plt.savefig('images/image-mask.png', dpi=300, bbox_inches='tight')
# # plt.show()

start_y = 200
dest_y = 328
start_x = 600
dest_x = 728

##  for saving figures
# plot_two_with_map(im_m, y_pred_img, im_i=im_i, title="Predikce metody LBP (šedotón) - SVM",
#                   save_name="./figures/LBP_grayscale_SVM-2iter_1024_correct_2")
#
# plot_two_with_map(im_m[start_x:dest_x, start_y:dest_y], y_pred_img[start_x:dest_x, start_y:dest_y], im_i=im_i[start_x:dest_x, start_y:dest_y],
#                   title="Predikce metody LBP (šedotón) - SVM, výřez",
#                   save_name="./figures/LBP_grayscale_SVM-2iter_128_correct_2")
#
# plot_two_with_map(im_m[start_x+32:dest_x-32, start_y+32:dest_y-32], y_pred_img[start_x+32:dest_x-32, start_y+32:dest_y-32], im_i=im_i[start_x+32:dest_x-32, start_y+32:dest_y-32],
#                   title="Predikce metody LBP (šedotón) - SVM, výřez",
#                   save_name="./figures/LBP_grayscale_SVM-2iter_64_correct_2")

# plt.show()
