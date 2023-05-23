import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

pretrained = False
weights = DeepLabV3_ResNet50_Weights.DEFAULT
num_classes = 8
image_num = 16 #  16-79, neodpovídá číslům snímků

# načtení modelu

saved_model_path = "./models/model_6.pth"

if pretrained:
    model3 = deeplabv3_resnet50(weights=weights)
    model3.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    nn.init.xavier_uniform_(model3.classifier[-1].weight)
    model3.load_state_dict(torch.load(saved_model_path))
    model3.eval()
else:
    model = deeplabv3_resnet50(num_classes=8)
    model.load_state_dict(torch.load(saved_model_path))

model.eval()

# transformace
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Paths, stačí měnit číslo obrázku do 63
# každá cesta obsahuje 16x4 obrázky
img_path = "../claassical/LoveDA_Test_16/Rural/images_png_512/"+image_num+".png"
mask_path = "../classical/LoveDA_Test_16/Rural/masks_png_512/"+image_num+".png"
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
pred = torch.argmax(prediction.squeeze(), dim=0)
mask_np = np.shape(pred.numpy())


mask_nd = np.asarray(Image.open(mask_path))



from methods import plot_two_with_map, plot_six_nn

plot_two_with_map(mask_nd, pred, im_i=Img_PIL, title="Predikce modelu "+saved_model_path)



# colors = ['#000000', '#666666', '#d22d04', '#840f8f', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# # descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
# myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=8)
#
# plt.imshow(mask, cmap=myCmap)
# plt.show()



# img = Image.fromarray(mask.numpy())
# img.show()
