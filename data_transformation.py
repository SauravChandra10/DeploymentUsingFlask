import numpy as np
import pandas as pd

df = pd.read_csv('Predict-24-jan-scenarioV1.csv')

print(df.head())

df['date']=pd.to_datetime(df['Date'], format = '%d-%m-%Y')
df['Year'] = df['date'].dt.year
df['Day'] = df['date'].dt.day

df=df.drop(['Date','date'],axis=1)

df.rename(columns={'CNY':'CNH'},inplace=True)

df.to_csv('test.csv', index=False)
