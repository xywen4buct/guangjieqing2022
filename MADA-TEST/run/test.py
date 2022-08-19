import torch
from dataload.DataLoader import get_data
from utils.get_metrics import get_metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.par import getpar
def test(source_db,db_name,model,alpha):
    target_data,target_label = get_data(db_name=db_name)

    scaler = MinMaxScaler()
    target_data = scaler.fit_transform(target_data)

    target_data = target_data.astype(np.float32)

    # print(model)
    target_data_tensor = torch.from_numpy(target_data)


    model.eval()

    pred_res = model(input_data=target_data_tensor,alpha=alpha)[1]

    pred_res_ = pred_res.detach().numpy()

    source_target_dict = getpar()

    for key,val in source_target_dict.items():
        if (source_db,db_name) == key:
            pred_res_ = np.where(pred_res_ > val, 1.0, 0.0)
            break
    # if db_name == "ant":
    #     pred_res_ = np.where(pred_res_>0.75,1.0,0.0)
    # else:
    #     pred_res_ = np.where(pred_res_>0.65,1.0,0.0)

    #print(pred_res)
    recall,f1_s,auc,fpr = get_metrics(y_true=target_label,y_pred=pred_res_,
                                      y_score=pred_res.detach().numpy())
    print(recall)
    print(f1_s)
    print(auc)
    print(fpr)