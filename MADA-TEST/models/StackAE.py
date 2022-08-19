import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable

class BaseAE(nn.Module):

    def __init__(self,layer_th,input_dim,hidden_dim):
        super(BaseAE,self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("encoder_{}".format(layer_th),nn.Linear(
            in_features=input_dim,out_features=hidden_dim
        ))
        self.encoder.add_module("activate_encoder_{}".format(layer_th),nn.Sigmoid())
        self.decoder = nn.Sequential()
        self.decoder.add_module("decoder_{}".format(layer_th),nn.Linear(
            in_features=hidden_dim,out_features=input_dim
        ))
        self.decoder.add_module("activate_decoder_{}".format(layer_th),nn.Sigmoid())

    def forward(self,input_data):
        encoder_out = self.encoder(input_data)
        decoder_out = self.decoder(encoder_out)
        return (encoder_out,decoder_out)

class StackAEs(nn.Module):

    def __init__(self,layer_list):
        super(StackAEs, self).__init__()
        self.layer_list = layer_list
        self.ae_list = []


    def get_noisy_inputs(self,inputs,corruption_ratio):
        random_num = torch.rand(size=inputs.shape,dtype=torch.float32,requires_grad=False)
        zeros = torch.zeros_like(random_num,dtype=torch.float32,requires_grad=False)
        # noisy_data = torch.where(random_num>corruption_ratio,random_num,zeros)
        noisy_data = inputs * (Variable(inputs.data.new(inputs.size()).normal_(0, 0.1)) > -.1).type_as(inputs)
        return noisy_data

    def forward(self,input_data,epochs):
        # input_data应当包含两部分，source和target，使两方面数据的信息都能被保留
        # 假定input_data为DataLoader
        for i in range(len(self.layer_list)-1):
            loss_fn = nn.MSELoss(reduction="mean")
            ae = BaseAE(layer_th=i+1,input_dim=self.layer_list[i],
                             hidden_dim=self.layer_list[i+1])
            ae.train()
            opt = optim.Adam(params=ae.parameters(),lr=1e-3)
            for epoch in range(epochs):
                for step,b_x in enumerate(input_data):
                    opt.zero_grad()
                    if len(self.ae_list) == 0:
                        b_x_noize = self.get_noisy_inputs(b_x,0.5)
                        _,decoder_out = ae(b_x_noize)
                    else:
                        for j in range(len(self.ae_list)):
                            b_x_noize = self.get_noisy_inputs(b_x,0.5)
                            self.ae_list[j].eval()
                            encoder_out,decoder_out = self.ae_list[j](b_x_noize)
                            b_x = encoder_out
                        _,decoder_out = ae(encoder_out)
                    loss = loss_fn(decoder_out,b_x)
                    print("model : {}, epoch : {}, step : {}, loss : {}".format(i,epoch,step,loss))
                    loss.backward()
                    opt.step()
            self.ae_list.append(ae)
        stack = nn.Sequential()
        stackDecoder = nn.Sequential()
        stackEncoder = nn.Sequential()
        for i in range(len(self.ae_list)):
            stack.add_module("encoder_model_{}".format(i),self.ae_list[i].encoder)
            stackEncoder.add_module("encoder_model_{}".format(i),self.ae_list[i].encoder)
        for i in range(len(self.ae_list)):
            stack.add_module("decoder_model_{}".format(i),self.ae_list[2-i].decoder)
            stackDecoder.add_module("decoder_model_{}".format(i),self.ae_list[2-i].decoder)
        return stack,stackEncoder,stackDecoder