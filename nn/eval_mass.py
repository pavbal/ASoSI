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
from methods import plot_two_with_map, plot_six_nn


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

num_classes = 8
weights = DeepLabV3_ResNet50_Weights.DEFAULT


image_nums = ["2696_2", "2869_0", "3520_3", "3619_0", "3650_0", "3709_3", "3910_3", "3987_0" , "4017_2" , "4185_3", "2607_0", "4068_3", "2914_3"]

# načtení modelů
saved_model_path_1 = "./models_chosen/model_14_maxi_512_35_SGD_eval_bestMiou_17.pth"
saved_model_path_2 = "./models_chosen/model_18_maxi_512_21_SGD_eval_bestMiou_10.pth"
saved_model_path_3 = "./models_chosen/model_22_maxi_512_20_SGD_eval_bestMiou_5.pth"

model1 = deeplabv3_resnet50(num_classes=8)
model1.load_state_dict(torch.load(saved_model_path_1))
model1.eval()

model2 = deeplabv3_resnet50(num_classes=8)
model2.load_state_dict(torch.load(saved_model_path_2))
model2.eval()


model3 = deeplabv3_resnet50(weights=weights)
model3.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
nn.init.xavier_uniform_(model3.classifier[-1].weight)
model3.load_state_dict(torch.load(saved_model_path_3))
model3.eval()

# model3 = deeplabv3_resnet50(num_classes=8)
# model3.load_state_dict(torch.load(saved_model_path_3))
# model3.eval()

# transformace
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for image_num in image_nums:

    img_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/images_png_512/"+image_num+".png"
    mask_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA/Val/Rural_and_Urban/masks_png_512/"+image_num+".png"

    # náhled obrázku a masky
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

    prediction = model1(torch.unsqueeze(img, 0))['out']
    pred1 = torch.argmax(prediction.squeeze(), dim=0)

    prediction = model2(torch.unsqueeze(img, 0))['out']
    pred2 = torch.argmax(prediction.squeeze(), dim=0)

    prediction = model3(torch.unsqueeze(img, 0))['out']
    pred3 = torch.argmax(prediction.squeeze(), dim=0)

    mask_nd = np.asarray(Image.open(mask_path))

    # plot_two_with_map(mask_nd, pred1, im_i=Img_PIL, title="Predikce modelu 1 na snímku "+image_num,
    #                   save_name="./figures/Model_1_ignore")
    # # plt.show()

    plot_six_nn(Img_PIL, mask_nd, pred1, pred2, pred3, title="Predikce modelů 7, 8 a 9 na snímku "+image_num,
                save_name="./figures/modely789-"+image_num,
                models_num=["7", "8", "10"],
                # image_num=image_num,
                show=False
                )

