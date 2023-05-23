import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from CustomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision import transforms
from methods import bce_loss, dice_loss, compute_iou, f1_score, iou
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import MulticlassJaccardIndex
from SegmentationLoss import SegmentationLoss

# from CustomDeepLabV3_2 import CustomDeepLabV3, CustomDeepLabv3_2

# #  trénovací skript

# cesty
base_path_train = "C:/Users/pavba/PycharmProjects/projekt-5/nn_baka/LoveDA_Train_16/Rural/"  #LoveDA_Train_16
base_path_train = "../LoveDA_Train_16/Rural/"  #LoveDA_Train_16
# base_path_train = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Train/Rural_and_Urban/" #  maxi datasetík
img_dir_train = base_path_train + "images_png_512/"
mask_dir_train = base_path_train + "masks_png_512/"

base_path_val = "C:/Users/pavba/PycharmProjects/projekt-5/nn_baka/LoveDA_Train_16/Rural/"
base_path_val = "../LoveDA_Train_16/Rural/"
# base_path_val = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/"  #  maxi datasetík

img_dir_val = base_path_val + "images_png_512/"
mask_dir_val = base_path_val + "masks_png_512/"

saved_model_epochs = 0
saved_model_path = ""

##  v případě dotrénovávání některého z modelů
# saved_model_path = "./models/model_8_maxi_512_30_adam.pth"
# saved_model_epochs = 30


## hyperparametry
num_classes = 8  # včetně pozadí a ignoruj
num_epochs_to_train = 35
batch_size = 4
alfa = 0.7
beta = 0.5

# cuda
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

cpu = torch.device("cpu")


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

# inicializace datasetů
train_dataset = CustomImageDataset(img_dir=img_dir_train, mask_dir=mask_dir_train,
                                   transform=transformsImg,
                                   target_transform=None) #předtím None # transformMask

eval_dataset = CustomImageDataset(img_dir=img_dir_val, mask_dir=mask_dir_val,
                                  transform=transformsImg,
                                  target_transform=None)

# inicializace dataloaderů
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# weights = DeepLabV3_ResNet50_Weights#.DEFAULT
model = deeplabv3_resnet50(num_classes=num_classes)
model_num = "14"
#
if saved_model_path != "":
    model.load_state_dict(torch.load(saved_model_path))

model.to(device)
# model.train()  #  přesunuto do cyklu

# ztrátová fce a optimizer
criterion_ce = nn.CrossEntropyLoss()
criterion_dice = Dice(num_classes=8)  #new
criterion_bce = nn.BCELoss()  #new

criterion = SegmentationLoss(alpha=alfa, beta=beta)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)

scheduler20 = MultiStepLR(optimizer, milestones=[17], gamma=0.1)  # model12
# scheduler_cond = MultiStepLR(optimizer, milestones=[1, 9], gamma=0.1)
tresh = 0.22
run_sched = False

IoU = JaccardIndex(task="multiclass", num_classes=8, ignore_index=0)
# IoU = MulticlassJaccardIndex(num_classes=8, ignore_index=0)

torch.cuda.empty_cache()

avg_val_loss = np.zeros(num_epochs_to_train)
avg_train_loss = np.zeros(num_epochs_to_train)
avg_val_accuracy = np.zeros(num_epochs_to_train)
avg_train_accuracy = np.zeros(num_epochs_to_train)
avg_f1_val = np.zeros(num_epochs_to_train)
mean_iou_val = np.zeros(num_epochs_to_train)

best_val_acc = 0
best_val_miou = 0
best_epoch_acc = 0
best_epoch_miou = 0

running_iou = 0.0

# Trénovací cyklus s průběžkou validací
for epoch in range(num_epochs_to_train):
    torch.cuda.empty_cache()
    correct = 0.0
    total = 0.0
    # model.to(device)
    model.train()
    running_loss_train = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, target = data#['image'], data['mask']
        inputs, target = inputs.to(device), target.to(device)
        # target = torch.argmax(target, dim=1)

        optimizer.zero_grad()

        outputs = model(inputs)['out']
        # mask = torch.argmax(outputs.squeeze(), dim=0)

        # binary_masks = [target.unsqueeze(1) == i for i in range(8)]
        # target2 = torch.cat(binary_masks, dim=1)
        # target2 = target2.float()
        # target2=target2.to(device)

        # loss = criterion_ce(outputs, target)
        loss = criterion_ce(outputs, target)# + alfa*criterion_bce(outputs, target) + beta*criterion_dice(outputs, target.int())  # changed
        # loss = criterion(outputs, target2)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item()

        values, pred = torch.max(outputs.data, 1)  # temp, delete later

        values, pred = torch.max(outputs.data, 1)
        total += (target != 0).sum().item()
        correct += (pred == target).sum().item() - (target == 0).sum().item()
    print('[%d] train_loss: %.3f' % (epoch + 1 + saved_model_epochs, running_loss_train / len(train_loader)))
    # TO DO
    #     running_loss někam pro následné vykreslení
    avg_train_loss[epoch] = running_loss_train / len(eval_loader)
    avg_train_accuracy[epoch] = 100 * correct / total

    # inputs, target = inputs.to(cpu), target.to(cpu)
    torch.cuda.empty_cache()
    # model.to(cpu)
    model.eval()
    with torch.no_grad():
        # validační smyčka
        running_loss_eval = 0.0
        correct = 0.0
        total = 0.0
        running_iou = 0.0  #new
        f1_val = 0.0
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

            f1_val += f1_score(outputs.cpu(), target.cpu())

            # loss = criterion_ce(outputs, target)
            loss = criterion_ce(outputs, target)# + alfa*bce_loss(outputs, target) + beta*dice_loss(outputs, target) # new

            running_loss_eval += loss.item()
            # nejsou zde loss.backwards() a optimizer.step()

        avg_val_loss[epoch] = running_loss_eval / len(eval_loader)
        avg_val_accuracy[epoch] = 100 * correct / total
        avg_f1_val[epoch] = f1_val / len(eval_loader)
        mean_iou_val[epoch] = running_iou / len(eval_loader)
        print('[%d] eval_loss: %.3f\t eval_accuracy: %.3f' % (epoch + 1 + saved_model_epochs, avg_val_loss[epoch], avg_val_accuracy[epoch]))
        torch.cuda.empty_cache()

    scheduler20.step()
    if mean_iou_val[epoch] > tresh:
        # scheduler_cond.step()
        run_sched = True

    if run_sched:
        # scheduler_cond.step()
        run_sched = True

    if epoch==0:
        best_val_acc = avg_val_accuracy[epoch] - 1
        best_val_miou = mean_iou_val[epoch] - 1
        best_val_loss = avg_val_loss[epoch] + 1

    if avg_val_accuracy[epoch] > best_val_acc:
        best_epoch_acc = epoch
        best_val_acc = avg_val_accuracy[epoch]
        best_model_weights_acc = model.state_dict()
        torch.save(best_model_weights_acc, './models_all/models5/model_' + model_num + '_maxi_512_' + str(
            num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestAcc_interrupted.pth')
    if mean_iou_val[epoch] > best_val_miou:
        best_epoch_miou = epoch
        best_val_miou = mean_iou_val[epoch]
        best_model_weights_miou = model.state_dict()
        torch.save(best_model_weights_miou, './models_all/models5/model_' + model_num + '_maxi_512_' + str(
            num_epochs_to_train + saved_model_epochs) + '_SGD_Zval_bestMiou_interrupted.pth')
    if avg_val_loss[epoch] < best_val_loss:
        best_epoch_loss = epoch
        best_val_loss = avg_val_loss[epoch]
        best_model_weights_loss = model.state_dict()
        torch.save(best_model_weights_loss, './models_all/models5/model_' + model_num + '_maxi_512_' + str(
            num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestLoss_interrupted.pth')

    torch.cuda.empty_cache()
    # if epoch > 0:
    #     if avg_val_accuracy[epoch-1] > avg_val_accuracy[epoch] and best_Acc:
    #         torch.save(model.state_dict(), './models/model_' + model_num + '_maxi_512_' + str(
    #             num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestAvgAcc.pth')
    #
    #     if mean_iou_val[epoch-1] > mean_iou_val[epoch] and best_IoU:
    #         torch.save(model.state_dict(), './models/model_' + model_num + '_maxi_512_' + str(
    #             num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestMeanIoU.pth')
    np.save("./models_all/metrics5/avg_val_acc_" + model_num + ".npy", avg_val_accuracy)
    np.save("./models_all/metrics5/avg_train_acc_" + model_num + ".npy", avg_train_accuracy)
    np.save("./models_all/metrics5/avg_val_loss_" + model_num + ".npy", avg_val_loss)
    np.save("./models_all/metrics5/avg_train_loss_" + model_num + ".npy", avg_train_loss)
    np.save("./models_all/metrics5/avg_val_f1_" + model_num + ".npy", avg_f1_val)
    np.save("./models_all/metrics5/mean_val_IoU_" + model_num + ".npy", mean_iou_val)

    torch.save(model.state_dict(), './models_all/models5/model_'+model_num+'_maxi_512_' + str(num_epochs_to_train + saved_model_epochs) + '_SGD_eval_interrupted.pth')

print('Finished Training')
torch.save(best_model_weights_acc, './models_all/models5/model_' + model_num + '_maxi_512_' + str(num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestAcc_'+str(best_epoch_acc)+'.pth')
torch.save(best_model_weights_miou, './models_all/models5/model_' + model_num + '_maxi_512_' + str(num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestMiou_'+str(best_epoch_miou)+'.pth')
torch.save(best_model_weights_loss, './models_all/models5/model_' + model_num + '_maxi_512_' + str(num_epochs_to_train + saved_model_epochs) + '_SGD_eval_bestLoss_'+str(best_epoch_loss)+'.pth')
torch.save(model.state_dict(), './models_all/models5/model_' + model_num + '_maxi_512_' + str(num_epochs_to_train + saved_model_epochs) + '_SGD_eval_maxEpochs.pth')
print("models saved")
# np.save("./metrics2/val_acc_"+model_num+".npy", avg_val_accuracy)
# np.save("./metrics2/train_acc_"+model_num+".npy", avg_train_accuracy)
# np.save("./metrics2/val_loss_"+model_num+".npy", avg_val_loss)
# np.save("./metrics2/train_loss_"+model_num+".npy", avg_train_loss)
# np.save("./metrics2/val_f1_"+model_num+".npy", avg_f1_val)
# np.save("./metrics2/mIoU"+model_num+".npy", mean_iou_val)
