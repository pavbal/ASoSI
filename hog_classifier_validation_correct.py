#vychází z hog_classifier
import os

import skimage
from sklearn import datasets
from skimage.feature import hog
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.image

import methods.methods
import metrics

# trenovaci data
data_matrix = np.load('./saved/dataset_hog_with_zeros.npy') # celý dataset
target_vect = np.ravel(np.load('./saved/mask_vect_with_zeros.npy')) # celý dataset
data_matrix = np.load('./saved/dataset_hog_with_zeros.npy') # celý dataset
target_vect = np.ravel(np.load('./saved/mask_vect_Train_with_zeros_2.npy')) # celý dataset

# data_matrix = np.load('./saved/dataset_hog_16.npy') # 16 snímků
# target_vect = np.ravel(np.load('./saved/mask_vect_hog_16.npy')) # 16 snímků

HOG_ORIENT = len(data_matrix[1,:]) #normálně 9
CELL_C = 16
IMAGE_LEN = 1024
hog_len = (int)(IMAGE_LEN/CELL_C)
dim = 9
global_counter_img = 0
max_iter = 10
print(np.shape(data_matrix))

# bayes
# gnb = sklearn.naive_bayes.GaussianNB(priors=None)
# gnb.fit(data_matrix, target_vect)
# print(gnb.n_features_in_)

# SVM
# max_iter=10
svc = svm.SVC(max_iter=max_iter)
svc.fit(data_matrix, target_vect)
print(svc.n_features_in_)

folder_dir_base = "./LoveDA/Val/" # celý dataset
file_name_list = sorted(os.listdir("./LoveDA/Val/Rural/images_png"), key=len) + sorted(os.listdir("./LoveDA/Val/Urban/images_png"), key=len)
NUMBER_IMAGES_VAL = len(file_name_list)

hog_scale = int(IMAGE_LEN/CELL_C)
dataset_hog_val = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES_VAL, HOG_ORIENT), dtype=float)

running_acc = 0.0
running_miou = 0.0
running_f1 = 0.0
all_pixels = 0
true_positive_pixels = 0
# true_negative_pixels = 0

for folder_level_1 in sorted(os.listdir(folder_dir_base), key=len):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in sorted(os.listdir(folder_dir_2), key=len):
        # ziskani obrazku a jeho masky
        file_name_image = folder_dir_base + folder_level_1 + "/images_png/" + image # celý dataset
        file_name_mask = folder_dir_base + folder_level_1 + "/masks_png/" + image # celý dataset
        img = skimage.io.imread(file_name_image)
        mask = skimage.io.imread(file_name_mask)

        cells_in_img = int(len(img) / CELL_C)*int(len(img) / CELL_C)

        # uprava masky pro hog
        scale_mask = int(len(mask) / CELL_C)
        shape_mask = (scale_mask, scale_mask)
        mask_hog = np.zeros(shape_mask, dtype=int)

        for x in range(len(mask_hog)): # jde o délku jedné strany, nikoliv počet prvků
            for y in range(len(mask_hog)):
                mezX1 = x * CELL_C
                mezX2 = mezX1 + CELL_C - 1
                mezY1 = y * CELL_C
                mezY2 = mezY1 + CELL_C - 1
                cell = mask[mezX1:mezX2, mezY1:mezY2]
                mask_hog[x, y] = st.mode(cell.flatten())

        # vytvareni vektoru masek
        mez1 = cells_in_img * global_counter_img
        mez2 = cells_in_img * global_counter_img + cells_in_img

        # vytvareni datasetu
        # fv = np.zeros(1024, 1024, 3)
        fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
                 visualize=False, channel_axis=2)
        fv = np.reshape(fv, (cells_in_img, HOG_ORIENT))
        # dataset_hog_val[mez1:mez2, 0:HOG_ORIENT] = fv

        # y_pred = gnb.predict(fv).reshape(-1, 1)
        y_pred = svc.predict(fv).reshape(-1, 1)
        y_pred_reshaped = np.reshape(y_pred, (hog_len, int(len(y_pred) / hog_len)))
        pred_mask = y_pred_reshaped
        # pred_mask = pred_mask.transpose()
        real_pred_mask = np.zeros((IMAGE_LEN, IMAGE_LEN))
        for y in range(0, hog_len):  # změněno pořadí
            for x in range(0, hog_len):
                real_pred_mask[CELL_C * x:CELL_C * (x + 1), CELL_C * y:CELL_C * (y + 1)] = pred_mask[x, y]

        # místo na kritéria
        no_zero_class = mask!=0
        zero_class = mask == 0
        all_pixels = np.sum(no_zero_class)
        true_positive_pixels = np.sum(np.logical_and(real_pred_mask==mask, no_zero_class))

        running_miou += metrics.miou(mask, real_pred_mask)
        running_acc += true_positive_pixels/all_pixels
        running_f1 += methods.methods.f1_score(mask, real_pred_mask)

        global_counter_img += 1
print(np.shape(running_acc))
print("average accuracy: ", running_acc/NUMBER_IMAGES_VAL)
print("mean IoU: ", running_miou/NUMBER_IMAGES_VAL)
print("average F1-score: ", running_f1/NUMBER_IMAGES_VAL)

# # testovaci data
# test_img_data = np.load('./saved/test_img_vect_hog.npy')
# test_mask = np.load('./saved/test_mask_hog.npy')
# mask_visual = np.reshape(test_mask, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))

# # originalni maska a obrazek pro vizualizaci
# img = skimage.io.imread("./LoveDA/Val/Rural/images_png/" + str(test_img_number) + ".png")
# mask = skimage.io.imread("./LoveDA/Val/Rural/masks_png/" + str(test_img_number) + ".png")


# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)




# y_pred = gnb.predict(dataset_hog_val).reshape(-1, 1)
# y_pred = svc.predict(test_img_data).reshape(-1, 1)  # SVM



# test_data_3d = np.zeros((1024, 1024, dim), dtype=int)
# test_target_3d = np.zeros((1024, 1024, 1), dtype=int)
#
#
# y_pred_reshaped = np.reshape(y_pred, (hog_len, int(len(y_pred)/hog_len)))
# for i in range(0, len(y_pred_reshaped[1,:]), hog_len):
#     pred_mask = y_pred_reshaped[0:hog_len, i:(i+hog_len)]
#     # pred_mask = pred_mask.transpose()
#     real_pred_mask = np.zeros((IMAGE_LEN, IMAGE_LEN))
#     for y in range(0, hog_len):  # změněno pořadí
#         for x in range(0, hog_len):
#             real_pred_mask[CELL_C*x:CELL_C*(x+1), CELL_C*y:CELL_C*(y+1)] = pred_mask[x,y]
#
#     # matplotlib.image.imsave('./Result/'+file_name_list[(int)(i/hog_len)], real_pred_mask)


# for x in range(0, (int)(len(target_vect)/(IMAGE_LEN)*CELL_C)):




# print("y_pred shape: ", y_pred.shape)
# print("test_mask shape: ", test_mask.shape)
# print("test_data shape: ", test_img_data.shape)
#
# print("Number of mislabeled points : %d" % np.sum(test_mask != y_pred))
#
# y_pred_img = np.reshape(y_pred, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))


# # # Vykresleni---------------------------------------------------------------------
# #
# # cmap1 = plt.cm.gray
#
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
