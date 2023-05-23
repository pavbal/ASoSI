import numpy as np
import matplotlib.pyplot as plt

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

# Plotting
plt.figure(figsize=(10, 6))
for class_idx in range(num_classes):
    plt.plot(range(1,num_epochs+1), iou_values[:, class_idx], color=colors[class_idx], label=descriptions[class_idx], linewidth=3)

plt.plot(range(1,num_epochs+1),ious, 'k', linewidth=5, label=descriptions[7])

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('IoU', fontsize=14)
plt.title('IoU per Class', fontsize=16)
plt.legend(loc='lower right')

# Set x-axis ticks as whole numbers from 1 to 20
plt.xticks(np.arange(1, num_epochs + 1))

# Set y-axis scale to a more dense scale
plt.locator_params(axis='y', nbins=20)

plt.show()