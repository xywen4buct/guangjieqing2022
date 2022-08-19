import yaml
import numpy as np
def getpar():

    res = dict()

    f = open("D:/transfer-learnings/MADA-TEST/utils/par.yaml", "r", encoding='utf-8')

    data = yaml.load(f, Loader=yaml.FullLoader)

    f.close()

    for key, val in data.items():
        for k, v in val.items():
            res[(key,k)] = eval(v)
    return res