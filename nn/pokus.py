import torch

from methods import compute_iou
from methods import compute_iou_per_class


input_tensor = torch.randn(224, 224)
tensor = torch.randint(low=0, high=8, size=(512, 512))

print(compute_iou(tensor, tensor))
print(compute_iou_per_class(tensor, tensor))