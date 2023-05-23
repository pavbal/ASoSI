from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import generic_filter
import numpy as np

def smooth_mask(mask):
    def replace_if_surrounded(arr):
        center = arr[len(arr) // 2]
        if np.sum(arr == center) >= 6:
            return center
        else:
            return arr[0]

    return generic_filter(mask, replace_if_surrounded, size=(3, 3))

# def plot_two_with_map(im_m, y_pred_img, im_i=None, mask_hog=None, title=None):
#     num_classes = 8
#     colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
#     descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
#     hog = False
#
#     myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
#     class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
#
#     if im_i is not None:
#         # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
#         if mask_hog is not None:
#             fig, (ax0, ax1, ax_h, ax2) = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1.080]})#, 'height_ratios': 1})
#             hog = True
#             ax_h.axis('off')
#             ax_h.imshow(mask_hog, cmap=myCmap, vmin=0, vmax=num_classes - 1)
#             ax_h.set_title('Maska pro HOG')
#         else:
#             fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.080]})#, 'height_ratios': 1})
#
#         ax0.axis('off')
#         ax0.imshow(im_i, cmap=plt.cm.gray)
#         ax0.set_title('Výchozí snímek')
#     else:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#
#     if title is not None:
#         fig.suptitle(title, fontsize=14, )
#     else:
#         fig.suptitle('Výsledek metody LBP', fontsize=14, y=0.98)
#
#
#     ax1.axis('off')
#     ax1.imshow(smooth_mask(im_m), cmap=myCmap, vmin=0, vmax=num_classes - 1)
#     ax1.set_title('Originální maska snímku')
#
#     ax2.axis('off')
#     im = ax2.imshow(y_pred_img, cmap=myCmap, vmin=0, vmax=num_classes - 1)
#     ax2.set_title('Predikovaná maska')
#
#     divider = make_axes_locatable(ax2)
#     cax = divider.append_axes('right', size='5%', pad=0.1)
#     cb = fig.colorbar(im, cax=cax, ticks=class_labels)
#
#     # navíc
#     ticks = (np.arange(len(descriptions)) + 0.5) / 1.14
#     cb.set_ticks(ticks)
#     cb.set_ticklabels(descriptions)
#     cb.ax.tick_params(labelsize=12)
#
#     cb.ax.set_title('Třídy', fontsize=12)
#
#     plt.show()

def plot_two_with_map(im_m, y_pred_img, im_i=None, mask_hog=None, title=None, save_name=None):
    num_classes = 8
    colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
    descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
    hog = False

    myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    if im_i is not None:
        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
        if mask_hog is not None:
            fig, (ax0, ax1, ax_h, ax2) = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1.080]})#, 'height_ratios': 1})
            hog = True
            ax_h.axis('off')
            ax_h.imshow(mask_hog, cmap=myCmap, vmin=0, vmax=num_classes - 1)
            ax_h.set_title('Maska pro HOG', fontdict={'fontsize': 20})
        else:
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.1), gridspec_kw={'width_ratios': [1, 1, 1.080]})#, 'height_ratios': 1})

        ax0.axis('off')
        ax0.imshow(im_i, cmap=plt.cm.gray)
        ax0.set_title('Výchozí snímek', fontdict={'fontsize': 20})
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if title is not None:
        fig.suptitle(title, fontsize=22, )
    else:
        fig.suptitle('Výsledek metody LBP', fontsize=20, y=0.98)


    ax1.axis('off')
    ax1.imshow(smooth_mask(im_m), cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax1.set_title('Originální maska snímku', fontdict={'fontsize': 20})

    ax2.axis('off')
    im = ax2.imshow(y_pred_img, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax2.set_title('Predikovaná maska', fontdict={'fontsize': 20})

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=class_labels)

    # navíc
    ticks = (np.arange(len(descriptions)) + 0.5) / 1.14
    cb.set_ticks(ticks)
    cb.set_ticklabels(descriptions, fontdict={'fontsize': 18})
    cb.ax.tick_params(labelsize=16) #12

    cb.ax.set_title('Třídy', fontsize=18)

    plt.margins(x=0, y=0)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


from sklearn.metrics import confusion_matrix
import numpy as np

import numpy as np

def compute_iou(y_pred, mask, num_classes=7):
    # toInt
    y_pred = torch.argmax(y_pred.squeeze(), dim=1)


    y_pred = np.array(y_pred, dtype=int)
    mask = np.array(mask, dtype=int)
    y_pred = y_pred.astype(int)
    mask = mask.astype(int)

    # Ignore 0
    mask[mask == 0] = num_classes + 1

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for c in range(1, num_classes + 1):
        y_pred_c = y_pred == c
        mask_c = mask == c
        intersection[c-1] = np.sum(np.logical_and(y_pred_c, mask_c))
        union[c-1] = np.sum(np.logical_or(y_pred_c, mask_c))

    # iou
    iou_score = np.sum(intersection[1:]) / np.sum(union[1:])

    return iou_score


import torch
import torch.nn.functional as F


def dice_loss(output, target, smooth=1e-7):
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss

def bce_loss(output, target):
    loss = F.binary_cross_entropy_with_logits(output, target)
    return loss


import torch

# def f1_score(y_pred, y_true, num_classes=7):
#     # Convert from one-hot to class index
#     y_pred = y_pred.argmax(dim=1)
#     y_true = y_true.argmax(dim=1)
#
#     # Ignore class 0
#     mask = y_true != 0
#     y_pred = y_pred[mask]
#     y_true = y_true[mask]
#
#     f1_scores = []
#     for c in range(1, num_classes+1):
#         true_positives = ((y_pred == c) & (y_true == c)).sum()
#         false_positives = ((y_pred == c) & (y_true != c)).sum()
#         false_negatives = ((y_pred != c) & (y_true == c)).sum()
#
#         precision = true_positives / (true_positives + false_positives + 1e-7)
#         recall = true_positives / (true_positives + false_negatives + 1e-7)
#
#         f1 = 2 * precision * recall / (precision + recall + 1e-7)
#         f1_scores.append(f1)
#
#     return np.array(f1_scores, dtype=np.float32)

def f1_score(y_pred, y_true, num_classes=7):
    # Convert from one-hot to class index
    y_pred = y_pred.argmax(dim=1)
    # y_true = y_true.argmax(dim=0)

    # Ignore class 0
    mask = y_true != 0
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    f1_scores = []
    for c in range(1, num_classes+1):
        true_positives = ((y_pred == c) & (y_true == c)).sum()
        false_positives = ((y_pred == c) & (y_true != c)).sum()
        false_negatives = ((y_pred != c) & (y_true == c)).sum()

        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / len(f1_scores)

    return avg_f1.item()

def iou(pred, target, n_classes = 8):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)


def compute_iou_per_class(pred, target, num_classes=8):
    pred[pred<0] = 0
    target[target<0] = 0
    """
    Computes the Intersection over Union (IoU) for a segmentation problem with 8 classes.

    Args:
        pred (torch.Tensor): Predicted segmentation mask of shape (N, H, W).
        target (torch.Tensor): Ground-truth segmentation mask of shape (N, H, W).
        num_classes (int): Number of classes in the segmentation problem.

    Returns:
        numpy array: IoU for each class of shape (num_classes - 1,).
    """

    # Ignore background class (index 0)
    pred = pred[:, 1:]
    target = target[:, 1:]
    num_classes -= 1

    ious = []
    inters = []
    unions = []
    for c in range(1, num_classes + 1):
        pred_c = pred[:, c - 1]
        target_c = target[:, c - 1]

        # Compute intersection and union
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection

        inters.append(intersection)
        unions.append(union)

        # Handle zero union exception
        if union == 0:
            ious.append(0)
        else:
            iou = intersection / union
            ious.append(iou)

    # return np.array(ious)
    return np.array(inters), np.array(unions)

def compute_iou_per_class_2(pred, target, num_classes=8):
    """
    Computes the Intersection over Union (IoU) for a segmentation problem with 8 classes.

    Args:
        pred (torch.Tensor): Predicted segmentation mask of shape (N, H, W).
        target (torch.Tensor): Ground-truth segmentation mask of shape (N, H, W).
        num_classes (int): Number of classes in the segmentation problem.

    Returns:
        numpy array: IoU for each class of shape (num_classes - 1,).
    """
    pred_1 = pred.view(-1)
    targ_1 = target.view(-1)
    # Ignore background class (index 0)
    # pred = pred[:, 1:, :, :]
    # target = target[:, 1:, :, :]
    num_classes -= 1

    # batch_size, height, width = pred.shape

    ious = []
    inters = []
    unions = []
    for c in range(1, num_classes + 1):
        pred_c = (pred_1 == c)
        target_c = (targ_1 == c)

        # Compute intersection and union
        intersection = (pred_c & target_c).sum(dim=0)
        union = pred_c.sum(dim=0) + target_c.sum(dim=0) - intersection

        inters.append(intersection)
        unions.append(union)

        # # Handle zero union exception
        # zero_union_mask = union == 0
        # iou = torch.zeros_like(intersection)
        # iou[~zero_union_mask] = intersection[~zero_union_mask] / union[~zero_union_mask]
        # ious.append(iou)

    return np.array(inters), np.array(unions)


def plot_six_nn(im_i, im_m, y2, y3, y4, title=None, save_name=None, models_num=["0","0","0"], image_num="", show=True):
    num_classes = 8
    colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
    descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
    hog = False

    myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(20, 4.3), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.09]})#, 'height_ratios': 1})


    if title is not None:
        fig.suptitle(title, fontsize=16, )
    else:
        fig.suptitle('Predikce', fontsize=18, y=0.98)

    ax0.axis('off')
    ax0.imshow(im_i, cmap=plt.cm.gray)
    ax0.set_title('Výchozí snímek '+image_num, fontdict={'fontsize': 16})

    ax1.axis('off')
    ax1.imshow(smooth_mask(im_m), cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax1.set_title('Originální maska', fontdict={'fontsize': 16})

    ax2.axis('off')
    im = ax2.imshow(y2, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax2.set_title('Predikce modelu '+models_num[0], fontdict={'fontsize': 16})

    ax3.axis('off')
    im = ax3.imshow(y3, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax3.set_title('Predikce modelu '+models_num[1], fontdict={'fontsize': 16})

    ax4.axis('off')
    im = ax4.imshow(y4, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax4.set_title('Predikce modelu '+models_num[2], fontdict={'fontsize': 16})

    # ax5.axis('off')
    # im = ax5.imshow(y5, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    # ax5.set_title('Predikce modelu ', fontdict={'fontsize': 20})

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=class_labels)

    # navíc
    ticks = (np.arange(len(descriptions)) + 0.5) / 1.14
    cb.set_ticks(ticks)
    cb.set_ticklabels(descriptions, fontdict={'fontsize': 16})
    cb.ax.tick_params(labelsize=14) #12

    cb.ax.set_title('Třídy', fontsize=16)

    plt.margins(x=0, y=0)

    fig.subplots_adjust(wspace=0.07)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
