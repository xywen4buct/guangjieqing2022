import pandas as pd
import pymysql

def get_data(db_name):

    connection = pymysql.Connect(host="localhost",port=3306,user="root",
                                 passwd="123",db=db_name)
    cursor = connection.cursor()

    sql = "select * from all_data_retain"

    rows = cursor.execute(sql)

    result = cursor.fetchall()

    columns = []
    for i in range(len(cursor.description)):
        columns.append(cursor.description[i][0])

    cursor.close()
    connection.close()

    df = pd.DataFrame(data=result, columns=columns)

    labels = df.label.values

    df.drop(columns=['id', 'label', 'annotators', 'parameter_number', 'parameter', 'methodName', 'className',
                     'filename','m_exp_std'], inplace=True)

    return (df.values, labels)