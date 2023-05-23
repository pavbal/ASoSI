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

# classes = ('IGNORE', 'Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural')
classes = ('0','1','2','3','4','5','6','7')
txt = '0 - IGNORE, 1 - Background, 2 - Building, 3 - Road, 4 - Water, 5 - Barren, 6 - Forest, 7 - Agricultural'
txt_cz = '0 - IGNORUJ, 1 - Pozadí, 2 - Budovy, 3 - Silnice, 4 - Voda, 5 - Pustina, 6 - Les, 7 - Agrikultura'
classes_numpy = np.arange(len(classes))
percent_arrays = np.empty([3,8])

# def plot_barchart_subplot(pole, envir, count):
#     performance = pole
#     plt.bar(classes_numpy, performance, align='center', alpha=0.5)
#     plt.subplot(1, 3, count)
#     plt.xticks(classes_numpy, classes)
#     if (count == 1):
#         plt.ylabel('Percentage of pixels')
#     # plt.xlabel('Classes')
#     plt.title(envir)

def plot_barchart_subplot(pole, envir, count, group):
    performance = pole
    percent_arrays[count-1, 0:8] = pole

    #plt.figure(figsize=(8.5, 6))
    plt.subplot(1, 3, count)
    fig.tight_layout(pad=1, w_pad=1)

    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)

    if (count == 1):
        plt.ylabel('Procento pixelů')
    if envir == "Rural":
        plt.title("Venkovské oblasti")
    elif envir == "Urban":
        plt.title("Městské oblasti")
    else:
        plt.title("Městské a venkovské oblasti")
    # plt.xlabel('Classes')



folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
group = "Val"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/" + group + "/"


global_counter = 0
plot_counter = 1

figsize = (11, 6)
figsize = (11, 3)

fig = plt.figure(figsize=figsize)
fig.suptitle('Procentuální podíl jednotlivých tříd - '+group, fontsize=14, )
fig.text(.5, .01,'TŘÍDY: '+ txt_cz, ha='center')
fig.tight_layout(pad=1, w_pad=1, rect=(10,1))
# fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, )

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"


    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    count = len(fnmatch.filter(os.listdir(folder_dir_2), '*.*'))
    pixels_classes = np.zeros((8,), dtype=int)


    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/"+group+"/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)

        # class_occur += [np.sum(im==0)>0, np.sum(im==1)>0, np.sum(im==2)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0]
        pixels_classes += [np.sum(im == 0), np.sum(im == 1), np.sum(im == 2), np.sum(im == 3),
                           np.sum(im == 4),
                           np.sum(im == 5), np.sum(im == 6),
                           np.sum(im == 7)]  # print(np.sum(im[:,:]==1))

        counter += 1
        global_counter += 1
    pixels_classes_percentage = (pixels_classes / sum(pixels_classes)) * 100
    plot_barchart_subplot(pixels_classes_percentage, folder_level_1, plot_counter, group)
    print("folder_level_1: "+folder_level_1 + ", plot_counter: " + str(plot_counter))
    plot_counter = plot_counter + 1
    if (global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10):
        pixels_classes_copy = pixels_classes
        pixels_classes_percentage_copy = pixels_classes_percentage

count_urban = len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*'))
count_rural = len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*'))
rural_mpy = np.multiply(pixels_classes_percentage_copy, count_rural)
urban_mpy = np.multiply(pixels_classes_percentage, count_urban)
global_percent = np.divide(np.add(rural_mpy, urban_mpy), (count_rural + count_urban))

plot_barchart_subplot(np.array(global_percent), "global", 3, group)

# plot_barchart_subplot(pixels_classes_percentage, "Rural", 1)
# plot_barchart_subplot(pixels_classes_percentage_copy, "Urban", 2)


# plt.savefig('barchart_subplots/barchart_subplot_percent_'+group+'1.png', bbox_inches='tight')
plt.savefig('barchart_subplots/barchart_subplot_percent_'+group+'.png')#, bbox_inches='tight')
occur_percent = (np.rint(percent_arrays)).astype(int)
# np.savetxt("csv/percent_arrays_"+group+".csv", percent_arrays, delimiter=",")

print(pixels_classes_percentage)
print(pixels_classes_percentage_copy)
print(global_percent)

# diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')