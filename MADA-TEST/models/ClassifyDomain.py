import torch
import torch.nn as nn
from utils.basemodel import BaseModel
from models.function import ReverseLayerF

# 需要将前面预训练好的编码器传入
class CompleteModel(nn.Module):

    def __init__(self,stack_encoder,stack_decoder,hidden_dim):
        super(CompleteModel,self).__init__()
        self.stack_encoder = stack_encoder
        self.stack_decoder = stack_decoder

        # 分类器
        nameListClf = ["clf_fc_1","clf_act_1","clf_normal_1","clf_drop_1","clf_fc_2",
                       "clf_act_2","clf_normal_2","clf_fc_3","clf_act_3",]
        self.classifier = BaseModel(hidden_dim=hidden_dim,nameList=nameListClf)

        # 全局域分类器
        nameListDm = ["dm_fc_1","dm_act_1","dm_normal_1","dm_drop_1","dm_fc_2",
                      "dm_act_2","dm_normal_2","dm_fc_3","dm_act_3"]
        self.domainClf = BaseModel(hidden_dim=hidden_dim,nameList=nameListDm)

        # 局部域分类器
        nameListDmPartOne = ["dm_o_fc_1","dm_o_act_1","dm_o_normal_1","dm_o_drop_1",
                             "dm_o_fc_2","dm_o_act_2","dm_o_normal_2","dm_o_fc_3",
                             "dm_o_act_3"]
        self.partDomainClfOne = BaseModel(hidden_dim=hidden_dim,nameList=nameListDmPartOne)
        nameListDmPartZero = ["dm_z_fc_1","dm_z_act_1","dm_z_normal_1","dm_z_drop_1",
                              "dm_z_fc_2","dm_z_act_2","dm_z_normal_2","dm_z_fc_3",
                              "dm_z_act_3"]
        self.partDomainClfZero = BaseModel(hidden_dim=hidden_dim,nameList=nameListDmPartZero)

    def forward(self,input_data,alpha):
        hidden_feature = self.stack_encoder(input_data)
        reverse_feature = ReverseLayerF.apply(hidden_feature, alpha)
        classOutput = self.classifier(hidden_feature)
        reconstruct = self.stack_decoder(hidden_feature)
        domainClfOutput = self.domainClf(reverse_feature)
        # 局部域的特征需要加权
        partOneDmFeature = torch.multiply(reverse_feature,torch.reshape(classOutput,(-1,1)))
        partZeroDmFeature = torch.multiply(reverse_feature,
                                           torch.reshape((torch.ones_like(classOutput)-classOutput),
                                                         (-1,1)))
        partOneDmClf = self.partDomainClfOne(partOneDmFeature)
        partZeroDmClf = self.partDomainClfZero(partZeroDmFeature)

        domainClfOutput = torch.clip(input=domainClfOutput,min=0.0,max=1.0)
        partOneDmClf = torch.clip(input=partOneDmClf,min=0.0,max=1.0)
        partZeroDmClf = torch.clip(input=partZeroDmClf,min=0.0,max=1.0)
        return (reconstruct,classOutput,domainClfOutput,partOneDmClf,partZeroDmClf)