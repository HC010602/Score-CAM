import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)            # 继承自BaseCAM，创建ScoreCAM类时输入model_dict，提取type layer_name arch的值

    def forward(self, input, class_idx=None, retain_graph=False):       # 覆盖BaseCAM的forward
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()   # 原图输入模型进行正向传播，得到原图输入模型后输出的各类别置信度，同时运行到目标层时触发hook，提取目标层的输出激活图

        if class_idx is None:
            predicted_class = logit.max(1)[-1]      # 预测类别的序号
            score = logit[:, logit.max(1)[-1]].squeeze()        # 预测类别的置信度 tensor(27.6874, device='cuda:0', grad_fn=<SqueezeBackward0>)
        else:
            predicted_class = torch.LongTensor([class_idx])     # 指定生成某一类别的Score-CAM图像
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit)                    # softmax默认按行计算

        if torch.cuda.is_available():               # 将各参数加载到显存上，使用GPU处理
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']     # 取出激活图，由BaseCAM中对目标层注册的hook得到
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)            # 取出第i张激活图，大小为b x 1 x u x v
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)  # 激活图上采样到 h x w
              
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())   # 归一化

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model_arch(input * norm_saliency_map)           # 上采样后的激活图重新输入模型得到新的各类别置信度
              output = F.softmax(output)
              score = output[0][predicted_class]        # 高亮区域后目标类别的预测置信度

              score_saliency_map +=  score * saliency_map       # 新的置信度*激活图，叠加到score_saliency_map中
                
        score_saliency_map = F.relu(score_saliency_map)     # 去除无关类别的负值
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:        # score_saliency_map全为0的情况
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        # print(score_saliency_map.shape) ->torch.Size([1, 1, 224, 224])

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):          # 将ScoreCAM类变为可调用的对象，返回input输入forward函数后的输出
        return self.forward(input, class_idx, retain_graph)