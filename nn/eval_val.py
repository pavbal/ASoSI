import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.models import get_weight
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from CustomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision import transforms
from methods import bce_loss, dice_loss, compute_iou, f1_score, iou, compute_iou_per_class, compute_iou_per_class_2
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import MulticlassJaccardIndex
from SegmentationLoss import SegmentationLoss

batch_size = 4
num_classes = 8
pretrained = False
weights = DeepLabV3_ResNet50_Weights.DEFAULT
# saved_model_path = "./models_all/models5/model_14_maxi_512_35_SGD_eval_bestMiou_17.pth"
saved_model_path = "./models_chosen/model_6_mini_512_10_adam.pth"

print(saved_model_path)


if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("CUDA is NOT available.")
    device = torch.device("cpu")


if pretrained:
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    model.load_state_dict(torch.load(saved_model_path))
else:
    model = deeplabv3_resnet50(num_classes=8)
    model.load_state_dict(torch.load(saved_model_path))

model.to(device)
model.eval()


base_path_val = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/"  #  maxi datasetík
base_path_val = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Train/Rural_and_Urban/"  #  maxi datasetík
base_path_val = "../LoveDA_Train_16/Rural/"


img_dir_val = base_path_val + "images_png_512/"
mask_dir_val = base_path_val + "masks_png_512/"

# transformace obrázku i masky
transformsImg = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transformMask=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor()])


eval_dataset = CustomImageDataset(img_dir=img_dir_val, mask_dir=mask_dir_val,
                                  transform=transformsImg,
                                  target_transform=None)

eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

criterion_ce = nn.CrossEntropyLoss()

IoU = JaccardIndex(task="multiclass", num_classes=8, ignore_index=0)
IoU_bin = JaccardIndex(task="binary", num_classes=8, ignore_index=0)

# avg_val_loss = 0
# avg_val_accuracy= 0


with torch.no_grad():
    torch.cuda.empty_cache()
    # validační smyčka
    running_loss_eval = 0.0
    correct = 0.0
    total = 0.0
    running_iou = 0.0  #new
    f1_val = 0.0
    running_inters = np.zeros(num_classes-1)
    running_unions = np.zeros(num_classes - 1)
    for i, data in enumerate(eval_loader, 0):
        inputs, target = data#['image'], data['mask']
        inputs, target = inputs.to(device), target.to(device)  # jen pro cudu

        outputs = model(inputs)['out']
        values, pred = torch.max(outputs.data, 1)
        total += (target != 0).sum().item()
        correct += (pred == target).sum().item() - (target == 0).sum().item()

        # running_iou += compute_iou(outputs.cpu(), target.cpu())  # new

        pred = pred.clone().detach()
        IoU.update(pred.cpu(), target.cpu())
        iou = IoU(pred.cpu(), target.cpu())
        running_iou += iou

        inters, unions = compute_iou_per_class_2(pred.cpu(), target.cpu())
        running_inters += inters
        running_unions += unions

        f1_val += f1_score(outputs.cpu(), target.cpu())

        binary_masks = [target.unsqueeze(1) == i for i in range(8)]
        target2 = torch.cat(binary_masks, dim=1)
        target2 = target2.float()
        target2 = target2.to(device)

        loss = criterion_ce(outputs, target)# + alfa*bce_loss(outputs, target) + beta*dice_loss(outputs, target) # new
        # loss = criterion(outputs, target2)

        running_loss_eval += loss.item()
        # nejsou zde loss.backwards() a optimizer.step()

    avg_val_loss= running_loss_eval / len(eval_loader)
    avg_val_accuracy = correct / total
    avg_f1_val= f1_val / len(eval_loader)
    # mean_iou_val = running_iou / len(eval_loader)
    mean_iou_per_class_val = np.reshape(np.divide(running_inters, running_unions), (1, num_classes-1))
    mean_iou_val = np.mean(mean_iou_per_class_val)

    print('loss:\t %.3f' % (avg_val_loss))
    print('acc:\t %.3f' % (avg_val_accuracy))
    print('f1:  \t %.3f' % (avg_f1_val))
    print('mIoU:\t %.3f' % (mean_iou_val))
    print('mIoU_pC:\t ', mean_iou_per_class_val)
    print(" ")
    print('IoU 1:\t %.3f' % (mean_iou_per_class_val[0,0]))
    print('IoU 2:\t %.3f' % (mean_iou_per_class_val[0,1]))
    print('IoU 3:\t %.3f' % (mean_iou_per_class_val[0,2]))
    print('IoU 4:\t %.3f' % (mean_iou_per_class_val[0,3]))
    print('IoU 5:\t %.3f' % (mean_iou_per_class_val[0,4]))
    print('IoU 6:\t %.3f' % (mean_iou_per_class_val[0,5]))
    print('IoU 7:\t %.3f' % (mean_iou_per_class_val[0,6]))




