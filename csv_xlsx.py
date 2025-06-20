import pandas as pd

data = pd.read_csv('train_data_sort.csv')
data.to_excel("train_data_sort.xlsx", index=False, header=False, engine="openpyxl")