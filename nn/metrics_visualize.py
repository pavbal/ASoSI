import numpy as np
import matplotlib.pyplot as plt

# Define the names of the arrays we want to plot
array_names = ["mIoU10", "train_acc_10", "train_loss_10", "val_acc_10", "val_f1_10", "val_loss_10"]
array_names = ["mean_val_IoU_11", "avg_train_acc_11", "avg_train_loss_11", "avg_val_acc_11", "avg_val_f1_11", "avg_val_loss_11"]
array_names = ["mean_val_IoU_12", "avg_train_acc_12", "avg_train_loss_12", "avg_val_acc_12", "avg_val_f1_12", "avg_val_loss_12"]
array_names = ["mean_val_IoU_13", "avg_train_acc_13", "avg_train_loss_13", "avg_val_acc_13", "avg_val_f1_13", "avg_val_loss_13"]
array_names = ["mean_val_IoU_14", "avg_train_acc_14", "avg_train_loss_14", "avg_val_acc_14", "avg_val_f1_14", "avg_val_loss_14"]
array_names = ["mean_val_IoU_23", "avg_train_acc_23", "avg_train_loss_23", "avg_val_acc_23", "avg_val_f1_23", "avg_val_loss_23"]



# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Loop over the array names and plot each one
for i, name in enumerate(array_names):
    # Load the array from the file
    filepath = f"./metrics3/{name}.npy"
    filepath = f"./models_all/metrics14/{name}.npy"

    arr = np.load(filepath)

    # Plot the array on the appropriate subplot
    row = i // 3
    col = i % 3
    axs[row, col].plot(arr)
    axs[row, col].set_title(name)

    # Set the x-axis label and tick marks
    axs[row, col].set_xlabel("epoch")
    axs[row, col].set_xticks(range(0, 20, 5))

# Add a suptitle for the entire figure
fig.suptitle("Metrics over 60 epochs")

# Show the plot
plt.show()