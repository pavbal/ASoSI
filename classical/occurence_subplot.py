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
occur_arrays = np.empty([3,8])


def plot_barchart_3_subplot(pole, envir, count, group):
    performance = pole
    occur_arrays[count-1, 0:8] = pole
    #plt.figure(figsize=(8.5, 6))
    plt.subplot(1, 3, count)
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    fig.tight_layout(pad=1, w_pad=1)
    if (count == 1):
        plt.ylabel('Procento obrázků')
        # plt.ylabel('Počet obrázků')
    # plt.xlabel('Classes')
    # plt.title(envir)
    if envir == "Rural":
        plt.title("Venkovské oblasti")
    elif envir == "Urban":
        plt.title("Městské oblasti")
    else:
        plt.title("Městské a venkovské oblasti")


folder_dir = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Train/Rural/images_png"
group = "Train"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/" + group + "/"

global_counter = 0
plot_counter = 1

figsize = (11, 3)

fig = plt.figure(figsize=figsize)
fig.suptitle('Počty obrázků s výskytem jednotlivých tříd - '+group, fontsize=14)
fig.text(.5, .01,'TŘÍDY: '+ txt_cz, ha='center')
fig.tight_layout(pad=1, w_pad=1, rect=(10,1))

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"


    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    count = len(fnmatch.filter(os.listdir(folder_dir_2), '*.*'))
    class_occur = np.zeros((8,), dtype=int)


    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/"+group+"/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)

        # class_occur += [np.sum(im==0)>0, np.sum(im==1)>0, np.sum(im==2)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0]
        class_occur += [np.any(im[:, :] == 0), np.any(im[:, :] == 1), np.any(im[:, :] == 2), np.any(im[:, :] == 3),
                        np.any(im[:, :] == 4), np.any(im[:, :] == 5), np.any(im[:, :] == 6), np.any(im[:, :] == 7)]

        counter += 1
        global_counter += 1
    image_count = len(os.listdir(folder_dir_2))
    plot_barchart_3_subplot(class_occur/image_count*100, folder_level_1, plot_counter, group) #ubrat lomeno
    plot_counter = plot_counter + 1

    if (global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10):
        class_occur_copy = class_occur
        image_count1 = image_count

plot_barchart_3_subplot(np.add(class_occur, class_occur_copy)/(image_count+image_count1)*100, 'global', 3, group)


plt.savefig('barchart_subplots/barchart_subplot_occur_'+group+'_p.png')#, bbox_inches='tight')
occur_arrays = (np.rint(occur_arrays)).astype(int)
# np.savetxt("csv/occur_arrays_"+group+".csv", occur_arrays, delimiter=",")

# diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')
