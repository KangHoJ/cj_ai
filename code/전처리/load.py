import zipfile
import pathlib
import os
import Path
import json
import pandas as pd

def unzip(path):
    path_list = path.rglob('*.zip')
    for paths in (path_list):
        with zipfile.ZipFile(paths, 'r') as zip_ref:
            zip_ref.extractall(str(paths)[:-4])
            os.remove(str(zip_ref))

unzip('C:/Users/User/document/etc study/cj ai 해커톤/01.데이터')

def to_dataframe(path):
    path_list = path.rglob('Training/01.원천데이터/*')
    df_columns =[]
    for paths in path_list:
        json_path = paths.glob('*.json')
        for j in json_path:
            with open(j,encoding='utf-8') as json_file:
                data = json.load(json_file)['sourceDataInfo']
                df_columns.append(data)
    df = pd.DataFrame(df_columns)
    return df

df = to_dataframe(Path('./01.데이터'))
df.to_csv('X_train.csv')