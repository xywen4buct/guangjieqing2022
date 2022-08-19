import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self,nameList,hidden_dim):
        super(BaseModel,self).__init__()
        self.classifier = nn.Sequential()
        self.classifier.add_module(nameList[0], nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim * 2))
        self.classifier.add_module(nameList[1], nn.ReLU(True))
        self.classifier.add_module(nameList[2], nn.BatchNorm1d(hidden_dim * 2, affine=True))
        self.classifier.add_module(nameList[3], nn.Dropout(p=0.2))
        self.classifier.add_module(nameList[4], nn.Linear(in_features=hidden_dim * 2,
                                                          out_features=hidden_dim))
        self.classifier.add_module(nameList[5], nn.ReLU(True))
        self.classifier.add_module(nameList[6], nn.BatchNorm1d(hidden_dim, affine=True))
        self.classifier.add_module(nameList[7], nn.Linear(in_features=hidden_dim,
                                                          out_features=1))
        self.classifier.add_module(nameList[8], nn.Sigmoid())

    def forward(self,input_data):
        return self.classifier(input_data)