import torch
import torch.nn as nn
from models.function import ReverseLayerF
from utils.basemodel import BaseModel

class CompleteModel(nn.Module):

    def __init__(self,layer_list,hidden_dim):
        super(CompleteModel,self).__init__()
        self.stack_encoder = nn.Sequential()
        self.stack_encoder.add_module("encoder_linear_1",nn.Linear(in_features=layer_list[0],
                                                            out_features=layer_list[1]))
        self.stack_encoder.add_module("encoder_act_1",nn.Sigmoid())
        self.stack_encoder.add_module("encoder_linear_2",nn.Linear(in_features=layer_list[1],
                                                                   out_features=layer_list[2]))
        self.stack_encoder.add_module("encoder_act_2",nn.Sigmoid())
        self.stack_encoder.add_module("encoder_linear_3",nn.Linear(in_features=layer_list[2],
                                                                   out_features=layer_list[3]))
        self.stack_encoder.add_module("encoder_act_3",nn.Sigmoid())
        self.stack_decoder = nn.Sequential()
        self.stack_decoder.add_module("decoder_linear_1",nn.Linear(in_features=layer_list[-1],
                                                                   out_features=layer_list[-2]))
        self.stack_decoder.add_module("decoder_act_1",nn.Sigmoid())
        self.stack_decoder.add_module("decoder_linear_2",nn.Linear(in_features=layer_list[2],
                                                                   out_features=layer_list[1]))
        self.stack_decoder.add_module("decoder_act_2",nn.Sigmoid())
        self.stack_decoder.add_module("decoder_linear_3",nn.Linear(in_features=layer_list[1],
                                                                   out_features=layer_list[0]))
        self.stack_decoder.add_module("decoder_act_3",nn.Sigmoid())

        # 分类器
        nameListClf = ["clf_fc_1","clf_act_1","clf_normal_1","clf_drop_1","clf_fc_2",
                       "clf_act_2","clf_normal_2","clf_fc_3","clf_act_3",]
        self.classifier = BaseModel(hidden_dim=hidden_dim,nameList=nameListClf)

        # 全局域分类器
        nameListDm = ["dm_fc_1","dm_act_1","dm_normal_1","dm_drop_1","dm_fc_2",
                      "dm_act_2","dm_normal_2","dm_fc_3","dm_act_3"]
        self.domainClf = BaseModel(hidden_dim=hidden_dim,nameList=nameListDm)
        # 局部域分类器
        nameListDmPart = ["dm_o_fc_1","dm_o_act_1","dm_o_normal_1","dm_o_drop_1",
                             "dm_o_fc_2","dm_o_act_2","dm_o_normal_2","dm_o_fc_3",
                             "dm_o_act_3"]
        self.partDomainClf = BaseModel(hidden_dim=hidden_dim,nameList=nameListDmPart)
        # nameListDmPartZero = ["dm_z_fc_1","dm_z_act_1","dm_z_normal_1","dm_z_drop_1",
        #                       "dm_z_fc_2","dm_z_act_2","dm_z_normal_2","dm_z_fc_3",
        #                       "dm_z_act_3"]
        # self.partDomainClfZero = BaseModel(hidden_dim=hidden_dim,nameList=nameListDmPartZero)

    def forward(self,input_data,alpha):
        # print(input_data)
        hidden_feature = self.stack_encoder(input_data)
        reverse_feature = ReverseLayerF.apply(hidden_feature, alpha)
        classOutput = self.classifier(hidden_feature)
        reconstruct = self.stack_decoder(hidden_feature)
        domainClfOutput = self.domainClf(reverse_feature)
        # 局部域的特征需要加权
        # print(hidden_feature)
        partDmFeature = torch.multiply(reverse_feature,torch.reshape(classOutput,(-1,1)))
        # partZeroDmFeature = torch.multiply(reverse_feature,
        #                                    torch.reshape((torch.ones_like(classOutput)-classOutput),
        #                                                  (-1,1)))
        partDmClf = self.partDomainClf(partDmFeature)
        # partZeroDmClf = self.partDomainClfZero(partZeroDmFeature)

        # domainClfOutput = torch.clip(input=domainClfOutput,min=0.0,max=1.0)
        # partOneDmClf = torch.clip(input=partOneDmClf,min=0.0,max=1.0)
        # partZeroDmClf = torch.clip(input=partZeroDmClf,min=0.0,max=1.0)
        return (reconstruct,classOutput,domainClfOutput,partDmClf)
        # return (reconstruct,classOutput,domainClfOutput,partOneDmClf,partZeroDmClf)