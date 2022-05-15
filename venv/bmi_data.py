import keras
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data=pd.read_csv('C:\\Users\\Vitam\\OneDrive\\Рабочий стол\\bmi.csv')
num_transform=StandardScaler()
cat_tranform=OneHotEncoder(handle_unknown='ignore')
y=data['Index']
X=data.drop(['Index'],axis=1)
num_features=X.select_dtypes(exclude=['object']).columns
cat_features=X.select_dtypes(exclude=['int','float64','float32']).columns
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,shuffle=True,test_size=0.2)
preprocessor=ColumnTransformer(transformers=[('num',num_transform,num_features),('cat',cat_tranform,cat_features)])
model=keras.Sequential([Dense(20,input_shape=(1,3),activation='tanh'),Dense(1,activation='linear')])
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mean_absolute_error'])
my_pipeline=Pipeline(steps=[('preproccesor',preprocessor),('model',model)])
history=my_pipeline.fit(X_train,y_train,model__batch_size=32,model__epochs=53,model__validation_split=0.2)
fig,ax1=plt.subplots(ncols=1,nrows=1)
ax1.plot(history.__getattribute__('steps')[1][1].history.history['loss'],label='train')
ax1.plot(history.__getattribute__('steps')[1][1].history.history['val_loss'],label='val')
y_pred=my_pipeline.predict(X_test)
print(y_test.values.reshape(-1,1))
print(mean_absolute_error(y_pred,y_test.values.reshape(-1,1)))
print(my_pipeline.predict(np.array([190,75,1]).reshape(1,-1)))
ax1.legend()
ax1.grid(True)
plt.show()


