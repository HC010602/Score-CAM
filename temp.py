import torch
import torchvision.models as models
from utils import *

vgg = models.vgg16(pretrained=True).eval()

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()

logit = vgg(input_).cuda()