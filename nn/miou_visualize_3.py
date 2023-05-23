import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the number of classes and epochs
num_classes = 7
num_epochs = 20

# Define the IoU values (replace this with your loaded numpy array)
# iou_values = np.random.rand(num_epochs, num_classes)
iou_values = np.load("models_all/metrics14/mean_val_IoU_per_class23.npy")
ious = np.load("models_all/metrics14/mean_val_IoU_23.npy")
num_classes = np.shape(iou_values)[1]
num_epochs = np.shape(iou_values)[0]

# Define colors and descriptions
colors = ['#666666', '#d22d04', '#ff6bfd', '#0575e6', '#994200', '#1a8f00', '#ffd724']
descriptions = ['Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura', 'mIoU']
myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)

# Plotting
fig, ax = plt.subplots(figsize=(9, 7))

# ax.set_aspect('equal')
# ax.set_xlim([1, 20])
# ax.set_ylim([0.1, 0.8])
# ax.axis('off')

for class_idx in range(num_classes):
    ax.plot(range(1, num_epochs + 1), iou_values[:, class_idx], color=colors[class_idx], label=descriptions[class_idx], linewidth=3)

ax.plot(range(1, num_epochs + 1), ious, 'k', linewidth=5, label=descriptions[7])

ax.set_xlabel('Epochy', fontsize=18)
ax.set_ylabel('IoU', fontsize=18)
ax.set_title('Vývoj kritérií IoU při trénování modelu 10', fontsize=20)
# ax.legend(loc='lower right')
ax.set_xlim(1, 20)

# Set x-axis ticks as whole numbers from 1 to 20
ax.set_xticks(np.arange(1, num_epochs + 1))

# Set y-axis scale to a more dense scale
ax.locator_params(axis='y', nbins=20)

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.3)

# Create a custom legend with colorbar-like appearance
class_labels = np.arange(num_classes)
ticks = (np.arange(len(descriptions) -1) + 0.5) / (len(descriptions) - 1)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=myCmap, norm=plt.Normalize(vmin=0, vmax=1)), cax=cax, ticks=ticks)

# cb.set_label('Třídy', fontsize=16)
cb.ax.set_title('Třídy', fontsize=20)

# Set custom tick positions and labels for the legend
cb.set_ticks(ticks)
cb.set_ticklabels(descriptions[:-1], fontdict={'fontsize': 18})
cb.ax.tick_params(labelsize=18)

plt.margins(x=0, y=0)

fig.set_size_inches(15, 7)
plt.savefig("./graphs/iou_graph_model_10", bbox_inches='tight')

plt.show()