import pymysql
import pandas as pd

def get_data(db_name):
    connection = pymysql.connect(host='localhost', port=3306, user='root', passwd='123', db=db_name)

    sql = "select * from all_data_retain"

    curse = connection.cursor()

    row = curse.execute(sql)

    result = curse.fetchall()

    columns = []
    for i in range(len(curse.description)):
        columns.append(curse.description[i][0])

    curse.close()
    connection.close()

    df = pd.DataFrame(data=result, columns=columns)

    labels = df.label.values

    df.drop(columns=['id', 'label', 'annotators', 'parameter_number', 'parameter', 'methodName', 'className',
                     'filename', 'm_exp_std'], inplace=True)

    return (df.values,labels)