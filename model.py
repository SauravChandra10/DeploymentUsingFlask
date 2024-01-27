import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('FX-SAMPLE-TRAIING-22JAN2024VER1.csv')

# coverting Date from string to date-time
df['date']=pd.to_datetime(df['Date'], format = '%Y-%m-%d')
df['Year'] = df['date'].dt.year
df['Day'] = df['date'].dt.day

# getting X and y
X=df.drop(['Date','date','INR'],axis=1)
y=df['INR']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# creating model
# rfOptimised = RandomForestRegressor(
#     n_estimators=400,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     max_depth=None,
#     bootstrap=False
# )

# rfOptimised.fit(X_train,y_train)

X_test.to_csv('X_test.csv', index=False)

# pickle.dump(rfOptimised,open('model.pkl','wb'))