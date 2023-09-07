# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import *
from cam.scorecam import *

# alexnet
alexnet = models.alexnet(pretrained=True).eval()      # 加载pytorch官方在ImageNet上预训练好的模型权重，并设置为evaluation模式，
# (对于一些含有batch normalization或者是Dropout层的模型来说，训练时的froward和验证时的forward有计算上是不同的，因此在前向传递过程中需要指定模型是在训练还是在验证。)
alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
alexnet_scorecam = ScoreCAM(alexnet_model_dict)       # 创建alexnet的ScoreCAM类

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')      # 以RGB模式读取图片
input_ = apply_transforms(input_image)      # 图片resize为224*224，将PIL转换为 (N x C x H x W) 的tensor并标准化
if torch.cuda.is_available():
  input_ = input_.cuda()      # 对数据进行.cuda()处理,可以将内存中的数据复制到GPU的显存中去，从而可以通过GPU来进行运算
predicted_class = alexnet(input_).max(1)[-1]      # alexnet(input_)得到一维的各类别置信度，.max(1)[-1]按行查找最大值，输出最大值的索引，相当于输出预测类别的序号

scorecam_map = alexnet_scorecam(input_)     # 调用ScoreCAM类，传入预处理后的图像，返回score_saliency_map
with torch.no_grad():
  basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path='alexnet.png')

# vgg
vgg = models.vgg16(pretrained=True).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))
vgg_scorecam = ScoreCAM(vgg_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = vgg(input_).max(1)[-1]

scorecam_map = vgg_scorecam(input_)
with torch.no_grad():
  basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path='vgg.png')

# resnet
resnet = models.resnet18(pretrained=True).eval()
resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
resnet_scorecam = ScoreCAM(resnet_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = resnet(input_).max(1)[-1]

scorecam_map = resnet_scorecam(input_)
with torch.no_grad():
  basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path='resnet.png')