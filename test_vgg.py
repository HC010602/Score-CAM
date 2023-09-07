import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import *
from cam.scorecam import *

# VGG16
vgg = models.vgg16(pretrained=True).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))
vgg_scorecam = ScoreCAM(vgg_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00000057.JPEG')

input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = vgg(input_).max(1)[-1]

# print(vgg(input_))

# print(predicted_class)

scorecam_map = vgg_scorecam(input_)
with torch.no_grad():
  basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path='vgg.png')