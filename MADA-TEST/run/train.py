import torch
from torch.utils.data import TensorDataset,DataLoader
from dataload.DataLoader import get_data
from torch import optim
from torch import nn
from models.loss import FocalLoss
# from models.ClassifyDomain import CompleteModel
from models.StackAE import StackAEs
from models.ModelClassify import CompleteModel
from run.test import test
import numpy as np
from sklearn.preprocessing import MinMaxScaler



cuda_iden = torch.cuda.is_available()
torch.manual_seed(111)
np.random.seed(111)
if cuda_iden:
    torch.cuda.manual_seed(111)

batch_size = 128
source_project = "kafka"
target_project = "tomcat"

source_data,source_label = get_data(db_name=source_project)
target_data,target_label = get_data(db_name=target_project)

scaler = MinMaxScaler()
source_data = scaler.fit_transform(source_data)
target_data = scaler.fit_transform(target_data)
# 转换为TENSOR
source_tensor = torch.from_numpy(source_data.astype(np.float32))
target_tensor = torch.from_numpy(target_data.astype(np.float32))

source_label_tensor = torch.from_numpy(source_label.astype(np.float32))
target_label_tensor = torch.from_numpy(target_label.astype(np.float32))

# 整理为torch标准数据
train_data = TensorDataset(source_tensor,source_label_tensor)
test_data = TensorDataset(target_tensor,target_label_tensor)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

len_dataloader = min(len(train_loader), len(test_loader))
data_source_iter = iter(train_loader)
data_target_iter = iter(test_loader)
layer_list = [58,116,58,50]
'''
# 先进行编码
all_tensor = torch.vstack([source_tensor,target_tensor])
all_loader = DataLoader(dataset=all_tensor,batch_size=batch_size,shuffle=False,num_workers=0)


stack_proxy = StackAEs(layer_list=layer_list)
stack,stackEncoder,stackDecoder = stack_proxy(all_loader,30)
# print(stack)

# 模型实例化
model = CompleteModel(stack_encoder=stackEncoder,stack_decoder=stackDecoder,
                      hidden_dim=layer_list[-1])
'''
model = CompleteModel(layer_list=layer_list,hidden_dim=layer_list[-1])

opt = optim.Adam(params=model.parameters(),lr=1e-5)

reconstruct_loss = nn.MSELoss(reduction="mean")
# 损失函数  Focal loss
loss_clf = FocalLoss(gamma=0.5,alpha=0.8)
# 全局域判断损失
all_domain_loss = nn.BCELoss(reduction="sum")
# 局部域判断损失 两种类别
part_domain_loss = nn.BCELoss(reduction="sum")
# part_zero_domain_loss = nn.BCELoss(reduction="sum")

# if cuda_iden:
#     model = model.cuda()
#     loss_clf = loss_clf.cuda()
#     all_domain_loss = all_domain_loss.cuda()
#     part_one_domain_loss = part_one_domain_loss.cuda()
#     part_zero_domain_loss = part_zero_domain_loss.cuda()

for p in model.parameters():
    p.require_grade = True

n_epoch = 50

print(model)


# 训练在CPU上
for epoch in range(n_epoch):

    len_dataloader = min(len(train_loader), len(test_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(test_loader)

    i = 0
    while i < len_dataloader-1:
        model.train()
        # alpha始终小于1
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 源域损失
        # 根据 全局域损失和局部域损失，得到omg，设置平衡参数lambda
        batch_data,batch_label = data_source_iter.next()

        opt.zero_grad()

        # 源域为0，目标域为1
        source_label_batch = torch.zeros_like(batch_label,dtype=torch.float32)
        # ,partOneDmClf,partZeroDmClf
        re,classOutput,domainClfOutput,partDmClf = model(batch_data,alpha)

        loss_reconstruct_source = reconstruct_loss(re,batch_data)

        lossClf = loss_clf(classOutput.view(-1,),batch_label)

        source_lossDm = all_domain_loss(domainClfOutput.view(-1,),source_label_batch)

        source_lossDm_part = part_domain_loss(partDmClf.view(-1,),source_label_batch)

        # source_lossDmZero = part_zero_domain_loss(partZeroDmClf.view(-1,),source_label_batch)

        # 目标域
        batch_data, batch_label = data_target_iter.next()

        target_label_batch = torch.ones_like(batch_label,dtype=torch.float32)
        # , partOneDmClf, partZeroDmClf
        re, _, domainClfOutput,partDmClf = model(batch_data, alpha)

        loss_reconstruct_target = reconstruct_loss(re,batch_data)

        target_lossDm = all_domain_loss(domainClfOutput.view(-1,),target_label_batch)

        target_lossDm_part = part_domain_loss(partDmClf.view(-1,),target_label_batch)

        # target_lossDmZero = part_zero_domain_loss(partZeroDmClf.view(-1,),target_label_batch)

        # 所有损失加
        Lg = (1./(batch_size*2)) * (source_lossDm + target_lossDm)
        LcAvg = (1./ (batch_size*2)) * (source_lossDm_part + target_lossDm_part)

        omg = Lg / (Lg + LcAvg)

        # loss_all = lossClf + Lg + LcAvg
        # 边缘分布  和  条件分布   的重要性   加大条件的，可以实现较高的Recall，但是FPR相应增高
        loss_all = lossClf + 0.5 * (omg * Lg + (1-omg) * LcAvg) + 0.5 * (loss_reconstruct_source + loss_reconstruct_target)
        # loss_all = lossClf + 0.5 * Lg + 0.5 * LcAvg
        # loss_all = lossClf + 0.5* (omg * Lg + (1-omg) * LcAvg)
        loss_all.backward()

        opt.step()

        # print("epoch: {}, step: {}, loss: {}, Lg: {}, Lc: {}, cl_loss:{}".format(
        #     epoch,i,loss_all,Lg,LcAvg,lossClf
        # ))
        print("epoch: {}, step: {}, loss: {}, Lg: {}, LcAvg:{}, cl_loss:{}".format(
            epoch, i, loss_all, Lg, LcAvg, lossClf
        ))

        i = i+1
    test(source_db=source_project,db_name=target_project,model=model,alpha=alpha)