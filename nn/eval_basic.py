import torch
from matplotlib.colors import LinearSegmentedColormap
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_num = "2522_0"
image_num = "2691_0"
image_num = "2696_2"  # good
image_num = "2737_0"
image_num = "2869_0"
image_num = "3493_2"
image_num = "3520_3"  # good
image_num = "3619_0"  # ne všechny původní masky jsou dokonalé
image_num = "3650_0"  #  good
image_num = "3709_3"  #  good
image_num = "3707_1"
image_num = "3850_3"
image_num = "3910_3"  # good
image_num = "3987_0"  # good
image_num = "4017_2"  # good
image_num = "4185_3"  # good, sus
image_num = "4068_3"
image_num = "2719_0"
image_num = "3910_3"

image_num = "4068_3"

# načtení modelu
saved_model_path = "./models_all/models6/model_15_maxi_512_30_SGD_eval_bestMiou_13.pth"
saved_model_path = "./models2/model_11_maxi_512_60_adam_eval_bestMiou_19.pth"
saved_model_path_1 = "./models/model_6_mini_512_10_adam.pth"
saved_model_path_2 = "./models/model_7_maxi_512_10_adam.pth"
saved_model_path_3 = "./models/model_8_maxi_512_30_adam.pth"
# saved_model_path = "./models/model_8_maxi_512_30_adam.pth"
model = deeplabv3_resnet50(num_classes=8)
model.load_state_dict(torch.load(saved_model_path))
model.eval()


model1 = deeplabv3_resnet50(num_classes=8)
model1.load_state_dict(torch.load(saved_model_path_1))
model1.eval()

model2 = deeplabv3_resnet50(num_classes=8)
model2.load_state_dict(torch.load(saved_model_path_2))
model2.eval()

model3 = deeplabv3_resnet50(num_classes=8)
model3.load_state_dict(torch.load(saved_model_path_3))
model3.eval()

# transformace
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Paths, stačí měnit číslo obrázku do 63
# každá cesta obsahuje 16x4 obrázky
img_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/images_png_512/"+image_num+".png"
mask_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/masks_png_512/"+image_num+".png"
# img_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA_Train_64/Rural/images_png_512/129.png"
# mask_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA_Train_64/Rural/masks_png_512/129.png"

# img_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Rural/images_png/4194.png"
# mask_path = "./Result/4194.png"


# náhled obrázku a masky ze stejného datasetu (model byl natrénován i na něm)
im = Image.open(img_path)
# im.show()
ma = Image.open(mask_path)
# ma.show()
# plt.imshow(ma, cmap=myCmap)

# img = data_transforms(img).unsqueeze(0)
img1 = read_image(img_path)
# img1 = torch.zeros(3, 512, 512)  #  hrozná to věc
img = data_transforms(img1)

Img_PIL = Image.open("C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/images_png_512/"+image_num+".png")
# Img_PIL = Image.open("C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Test/Rural/images_png/4191.png")

# Img_PIL = Image.new('L', (512, 512))

# vizualizace
# with torch.no_grad():
prediction = model(torch.unsqueeze(img, 0))['out']
print(np.shape(prediction))
mask = torch.argmax(prediction.squeeze(), dim=0)
print(np.shape(mask))
mask_np = np.shape(mask.numpy())
print(mask)
print(mask_np)
# to_pil_image(mask).show()¨¨


prediction = model1(torch.unsqueeze(img, 0))['out']
pred1 = torch.argmax(prediction.squeeze(), dim=0)

prediction = model2(torch.unsqueeze(img, 0))['out']
pred2 = torch.argmax(prediction.squeeze(), dim=0)

prediction = model3(torch.unsqueeze(img, 0))['out']
pred3 = torch.argmax(prediction.squeeze(), dim=0)


mask_nd = np.asarray(Image.open(mask_path))

# print("6: ",np.sum(mask_nd==6))
# print("5: ",np.sum(mask_nd==5))
# print("4: ",np.sum(mask_nd==4))
# print("3: ",np.sum(mask_nd==3))
# print("2: ",np.sum(mask_nd==2))
# print("1: ",np.sum(mask_nd==1))
# print("0: ",np.sum(mask_nd==0))




from methods import plot_two_with_map, plot_six_nn

# plot_two_with_map(mask_nd, pred1, im_i=Img_PIL, title="Predikce modelu 1 na snímku "+image_num,
#                   save_name="./figures/Model_1_ignore")
# # plt.show()

plot_six_nn(Img_PIL, mask_nd, pred1, pred2, pred3, mask, title="Predikce modelů 1, 2 a 3 na snímku "+image_num,
            save_name="./figures/modely123-"+image_num,
            models_num=["1", "2", "3"],
            # image_num=image_num,
            show=False
            )


# colors = ['#000000', '#666666', '#d22d04', '#840f8f', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# # descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
# myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=8)
#
# plt.imshow(mask, cmap=myCmap)
# plt.show()



# img = Image.fromarray(mask.numpy())
# img.show()
