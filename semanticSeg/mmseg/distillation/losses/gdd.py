import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_gdd (float, optional): Weight of distill_loss. Defaults to 0.00002
        temp_gdd (float, optional): Temperature of distillation. Defaults to 4
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 alpha_gdd=145,
                 temp_gdd=4,
                 ):
        super(FeatureLoss, self).__init__()
        self.alpha_gdd = alpha_gdd
        self.temp_gdd = temp_gdd
        self.name = name
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))
        self.normalize = ChannelNorm()

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.cal_distill_loss(preds_S, preds_T)*self.alpha_gdd
            
        return loss

    def cal_distill_loss(self, preds_S, preds_T):
        loss_kl = nn.KLDivLoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        # Gaussian noise
        mean, std = 0, 1
        gaus_noise = torch.normal(mean=mean, std=std, size=(N,1,H,W)).to(device)
        gaused_fea = torch.add(preds_S, gaus_noise)
        
        new_fea = self.generation(gaused_fea)
        
        norm_s = self.normalize(new_fea/self.temp_gdd)
        norm_t = self.normalize(preds_T.detach()/self.temp_gdd)

        norm_s = norm_s.log()
        distill_loss = loss_kl(norm_s, norm_t)
        distill_loss /= N * C
        return distill_loss * (self.temp_gdd ** 2)
    
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap