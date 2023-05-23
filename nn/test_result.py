import os

import matplotlib.image
import torch
import torchvision.transforms
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from methods import plot_two_with_map

from nn.CustomImageDataset import CustomImageDataset
from nn.TestImageDataset import TestImageDataset

batch_size = 4
pretrained = True
num_classes = 8
weights = DeepLabV3_ResNet50_Weights.DEFAULT


if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

cpu = torch.device("cpu")



# načtení modelu
saved_model_path = "./models_all/models14/model_23_maxi_512_20_SGD_eval_epoch_7.pth"
# saved_model_path = "./models/model_8_maxi_512_30_adam.pth"


if pretrained:
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    model.load_state_dict(torch.load(saved_model_path))
else:
    model = deeplabv3_resnet50(num_classes=8)
    model.load_state_dict(torch.load(saved_model_path))


# model = deeplabv3_resnet50(num_classes=8)
# model.load_state_dict(torch.load(saved_model_path))
model.to(device)
model.eval()

# transformace
transformsImg = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

toPIL = torchvision.transforms.ToPILImage

# Paths to valdir
img_dir_val = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Rural_and_Urban/images_png_512/"
mask_dir_val = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Rural_and_Urban/masks_png_512/"

test_dataset = TestImageDataset(img_dir=img_dir_val,
                                  transform=transformsImg
                                )

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

file_name_list = sorted(os.listdir("C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Rural/images_png"), key=len)\
                 + sorted(os.listdir("C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Urban/images_png"), key=len)

real_pred_mask = np.zeros((1024, 1024))
global_counter = 0
with torch.no_grad():
    for i, (inputs) in enumerate(test_loader):
        inputs = inputs.to(device)
        # print(inputs)
        outputs = model(inputs)["out"]
        _, pred = torch.max(outputs.data, 1)
        pred[pred!=0] = pred[pred!=0] - 1
        im0 = pred[0]
        im1 = pred[1]
        im2 = pred[2]
        im3 = pred[3]
        img1 = torch.cat((im0, im1), dim=1) #dim changed
        img2 = torch.cat((im2, im3), dim=1) #dim changed
        img = torch.cat((img1, img2), dim=0) #dim changed
        pred_mask = img.cpu().detach().numpy()
        # plot_two_with_map(pred_mask, pred_mask)
        # print(pred_mask.shape)
        # print(np.sum(pred_mask==6))
        pred_mask = np.uint8(pred_mask)
        real_pred_mask = to_pil_image(pred_mask, mode='L')
        # real_pred_mask = Image.fromarray(pred_mask, mode='L')

        real_pred_mask.save('./Result/' + file_name_list[global_counter])
        # matplotlib.image.imsave('./Result/' + file_name_list[global_counter], pred_mask)
        global_counter +=1


# plot_two_with_map(pred_mask, pred_mask)



# img = data_transforms(img).unsqueeze(0)
# img1 = read_image(img_path)
# img = transformsImg(img1)

Img_PIL = Image.open("C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/images_png_512/2522_0.png")

# # vizualizace
# # with torch.no_grad():
# prediction = model(torch.unsqueeze(img, 0))['out']
# print(np.shape(prediction))
# mask = torch.argmax(prediction.squeeze(), dim=0)
# print(np.shape(mask))
# mask_np = np.shape(mask.numpy())
# print(mask)
# print(mask_np)
# # to_pil_image(mask).show()
#
# # plt.plot(mask.numpy())
# # plt.show()
# mask_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/masks_png_512/2522_0.png"
# mask_nd = np.asarray(Image.open(mask_path))
#
# from methods import plot_two_with_map
#
# plot_two_with_map(mask_nd, mask, im_i=Img_PIL, title="DeepLabV3, Resnet50, 20 epoch, 512 px, validační dataset")
# plt.show()
#
# # colors = ['#000000', '#666666', '#d22d04', '#840f8f', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# # # descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
# # myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=8)
# #
# # plt.imshow(mask, cmap=myCmap)
# # plt.show()
#
#
#
# # img = Image.fromarray(mask.numpy())
# # img.show()