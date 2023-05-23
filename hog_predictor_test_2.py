#vychází z hog_classifier
import os
from PIL import Image
import skimage
from skimage.io import imsave
from sklearn import datasets
from skimage.feature import hog
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.image
import scipy.misc
import imageio

# trenovaci data
# data_matrix = np.load('./saved/dataset_hog_with_zeros.npy') # celý dataset
# target_vect = np.ravel(np.load('./saved/mask_vect_with_zeros.npy')) # celý dataset
data_matrix = np.load('./saved/dataset_hog_Train_with_zeros_2.npy') # celý dataset
target_vect = np.ravel(np.load('./saved/mask_vect_Train_with_zeros_2.npy'))  # celý dataset
data_matrix = np.load('./saved/dataset_hog_16.npy')  # 16 snímků
target_vect = np.ravel(np.load('./saved/mask_vect_hog_16.npy'))  # 16 snímků
data_matrix = np.load('./saved/dataset_hog_cell8.npy')  # 8cell
target_vect = np.ravel(np.load('./saved/mask_vect_hog_cell8.npy'))  # 8cell


# data_matrix = np.transpose(data_matrix)
# target_vect = np.transpose(target_vect)

print(data_matrix.shape)
print(target_vect.shape)

HOG_ORIENT = len(data_matrix[1,:]) #normálně 9
# HOG_ORIENT = len(data_matrix[:,1])
CELL_C = 16
CELL_C = 8
IMAGE_LEN = 1024
hog_len = (int)(IMAGE_LEN/CELL_C)
dim = 9
test_img_number = 3200
global_counter_img = 0

folder_dir_base = "./LoveDA/Test/" # celý dataset
file_name_list = sorted(os.listdir("./LoveDA/Test/Rural/images_png"), key=len) + sorted(os.listdir("./LoveDA/Test/Urban/images_png"), key=len)
NUMBER_IMAGES_TEST = len(file_name_list)

hog_scale = int(IMAGE_LEN/CELL_C)
dataset_hog_val = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES_TEST, HOG_ORIENT), dtype=float)

# bayes
gnb = sklearn.naive_bayes.GaussianNB(priors=None)
gnb.fit(data_matrix, target_vect)
print(gnb.n_features_in_)

# # VMS
# # max_iter=10
# svc = svm.SVC(max_iter=10)
# svc.fit(data_matrix, target_vect)
# print(svc.n_features_in_)


# y_pred = svc.predict(test_img_data).reshape(-1, 1)  # SVM

# cmap = plt.cm.get_cmap('gray', 7)  # 7 levels including 0




for folder_level_1 in sorted(os.listdir(folder_dir_base), key=len):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in sorted(os.listdir(folder_dir_2), key=len):
        # ziskani obrazku a jeho masky
        file_name_image = folder_dir_base + folder_level_1 + "/images_png/" + image # celý dataset
        img = skimage.io.imread(file_name_image)

        cells_in_img = int(len(img) / CELL_C)*int(len(img) / CELL_C)

        # vytvareni vektoru masek
        mez1 = cells_in_img * global_counter_img
        mez2 = cells_in_img * global_counter_img + cells_in_img

        # vytvareni datasetu
        # fv = np.zeros(1024, 1024, 3)
        fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
                 visualize=False, channel_axis=2)
        fv = np.reshape(fv, (cells_in_img, HOG_ORIENT))

        y_pred = gnb.predict(fv)#.reshape(-1, 1)
        # y_pred = svc.predict(fv)
        y_pred_reshaped = np.reshape(y_pred, (hog_len, hog_len)).astype('int32')

        # odečtení indexu třídy pro validacui na codalab:
        y_pred_reshaped[y_pred_reshaped != 0] = y_pred_reshaped[y_pred_reshaped != 0] - 1

        # if np.sum(y_pred!=0)>0:
        # print(np.sum(y_pred!=0))

        real_pred_mask = y_pred_reshaped.repeat(CELL_C, axis=0).repeat(CELL_C, axis=1)
        # print(real_pred_mask)
        # print(real_pred_mask!=0)
        real_pred_mask_img = Image.fromarray(real_pred_mask, mode='I')

        real_pred_mask_img.save('./Result/' + file_name_list[global_counter_img])
        # matplotlib.image.imsave('./Result/' + file_name_list[(int)(global_counter_img)], real_pred_mask)
        if image == "4191.png":
            real_pred_mask_img.show()
        # imsave('./Result/' + file_name_list[global_counter_img], real_pred_mask.astype(np.uint8))



        global_counter_img += 1


# testovaci data
test_img_data = np.load('./saved/test_img_vect_hog_8.npy')
test_mask = np.load('./saved/test_mask_hog_8.npy')
mask_visual = np.reshape(test_mask, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))

# originalni maska a obrazek pro vizualizaci
img = skimage.io.imread("./LoveDA/Val/Rural/images_png/" + str(test_img_number) + ".png")
mask = skimage.io.imread("./LoveDA/Val/Rural/masks_png/" + str(test_img_number) + ".png")


# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)


# y_pred_reshaped = np.reshape(y_pred, (hog_len, int(len(y_pred)/hog_len)))
# for i in range(0, len(y_pred_reshaped[1,:]), hog_len):
#     pred_mask = y_pred_reshaped[0:hog_len, i:(i+hog_len)]
#     # pred_mask = pred_mask.transpose()
#     real_pred_mask = np.zeros((IMAGE_LEN, IMAGE_LEN))
#     for y in range(0, hog_len):  # změněno pořadí
#         for x in range(0, hog_len):
#             real_pred_mask[CELL_C*x:CELL_C*(x+1), CELL_C*y:CELL_C*(y+1)] = pred_mask[x,y]
#
#     matplotlib.image.imsave('./Result/'+file_name_list[(int)(i/hog_len)], real_pred_mask)


# for x in range(0, (int)(len(target_vect)/(IMAGE_LEN)*CELL_C)):




print("y_pred shape: ", y_pred.shape)
print("test_mask shape: ", test_mask.shape)
print("test_data shape: ", test_img_data.shape)

print("Number of mislabeled points : %d" % np.sum(test_mask != y_pred))

y_pred_img = np.reshape(y_pred, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))


# # Vykresleni---------------------------------------------------------------------
#
# cmap1 = plt.cm.gray

from methods.methods import plot_two_with_map

plot_two_with_map(mask, y_pred_img, mask_hog=mask_visual, im_i=img, title="Test Hog")

# num_classes = 8
# colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# myCmap = ListedColormap(colors)
# class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
# bounds = np.arange(len(class_labels) + 1)
# norm = BoundaryNorm(bounds, len(class_labels))
#
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
# # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Výchozí snímek')
#
# ax2.axis('off')
# ax2.imshow(mask, cmap=myCmap, norm=norm)
# ax2.set_title('Maska snímku')
#
# # im_hybrid = np.hstack((im_m, y_pred_img))
#
# ax3.axis('off')
# ax3.imshow(mask_visual, cmap=myCmap, norm=norm)
# ax3.set_title('Maska pro HOG')
#
# ax4.axis('off')
# ax4.imshow(y_pred_img, cmap=myCmap, norm=norm)
# ax4.set_title('Predikovaná maska klasifikátorem GNB')
#
# plt.savefig('results2_visualisation/hog_result.png')
# plt.show()
