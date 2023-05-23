import skimage
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt


img = skimage.io.imread("./LoveDA/Val/Urban/images_png/3600.png")

img = skimage.data.astronaut()

_, fv = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                 visualize=True, channel_axis=2)

print(fv.shape)

from methods.methods import plot_two

plot_two(img, fv, title="Ukázka transformace obrázku pomocí HOG", save_name="./saved/HOG_visualize_astronaut.png")

img_g = skimage.color.rgb2gray(img)
lbp = local_binary_pattern(img_g, 8, 1, "uniform")

plot_two(img, lbp, title="Ukázka transformace obrázku pomocí LBP", save_name="./saved/LBP_visualize_astronaut.png")

# plt.figure
# plt.imshow(img)
# plt.show
# plt.imshow(fv)
