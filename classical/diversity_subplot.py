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

classes = ('1', '2', '3', '4', '5', '6', '7', '8')
classes_numpy = np.arange(len(classes))
plot_counter = 1
diverse_arrays = np.empty([3,8])


def plot_barchart_2_subplot(pole, envir, count, group):
    performance = pole
    #plt.figure(figsize=(8.5, 6))
    diverse_arrays[count-1, 0:8] = pole
    plt.subplot(1, 3, count)
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    fig.tight_layout(pad=1, w_pad=1)
    if(count == 1):
        plt.ylabel('Počet obrázků')

    # plt.title(envir)
    if envir == "Rural":
        plt.title("Venkovské oblasti")
    elif envir == "Urban":
        plt.title("Městské oblasti")
    else:
        plt.title("Městské a venkovské oblasti")



folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
group = "Train"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/" + group + "/"

diverse_classes_global = np.zeros((1366 + 1156,), dtype=int)
global_counter = 0

figsize = (11, 3)

fig = plt.figure(figsize=figsize)
fig.suptitle('Počety tříd na jednotlivých obrázcích - '+group, fontsize=14)
fig.text(.5, .01,'Počet tříd', ha='center')
fig.tight_layout(pad=1, w_pad=1, rect=(10,1))

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"

    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    count = len(fnmatch.filter(os.listdir(folder_dir_2), '*.*'))

    diverse_classes = np.zeros((count,), dtype=int)


    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/"+group+"/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)
        diverse_classes[counter] = len(np.unique(im))
        diverse_classes_global[global_counter] = len(np.unique(im))
        counter += 1
        global_counter += 1

    # print(pixels_classes)
    diverse_images = np.zeros((8,), dtype=int)
    diverse_images = [np.sum(diverse_classes == 1), np.sum(diverse_classes == 2),
                      np.sum(diverse_classes == 3), np.sum(diverse_classes == 4),
                      np.sum(diverse_classes == 5), np.sum(diverse_classes == 6),
                      np.sum(diverse_classes == 7), np.sum(diverse_classes == 8)]
    if global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10:
        diverse_images_copy = diverse_images

    plot_barchart_2_subplot(diverse_images, folder_level_1, plot_counter, group)
    plot_counter = plot_counter + 1
diverse_images_global = np.add(diverse_images, diverse_images_copy)
plot_barchart_2_subplot(diverse_images_global, 'global', 3, group)

plt.savefig('barchart_subplots/barchart_subplot_diverse_'+group+'.png')#, bbox_inches='tight')
diverse_arrays_arrays = (np.rint(diverse_arrays)).astype(int)
# np.savetxt("csv/diverse_arrays_"+group+".csv", diverse_arrays, delimiter=",")

##diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')
