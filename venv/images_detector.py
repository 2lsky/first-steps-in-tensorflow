import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense,Flatten,Dropout
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255
x_test=x_test/255
y_train_cat=keras.utils.to_categorical(y_train,10)
y_test_cat=keras.utils.to_categorical(y_test,10)
model=keras.Sequential([Flatten(input_shape=(28,28,1)),Dense(128,activation='relu'),Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train_cat,batch_size=32,epochs=5,validation_split=0.2)
fig,(ax1,ax2)=plt.subplots(ncols=2,nrows=1,figsize=(10,10))
ax1.plot(history.history['val_loss'],label='val_loss')
ax1.plot(history.history['loss'],label='train_loss')
ax2.plot(history.history['val_accuracy'],label='val_accuracy')
ax2.plot(history.history['accuracy'],label='train_accuracy')
ax1.legend()
ax2.legend()
plt.show()
model.evaluate(x_test,y_test_cat)
pred=model.predict(x_test)
pred_numbers=[]
false_test_positions=pd.DataFrame(columns=['real','pred'])
false_test_positions.index.names=['Number of test']
for sample in pred:
    pred_numbers.append(np.argmax(sample))
for i in range(0,len(pred_numbers)):
    if pred_numbers[i]!=y_test[i]:
        false_test_positions.loc[i]=pd.Series([y_test[i],pred_numbers[i]],index=['real','pred'])
for i in false_test_positions.index[:5]:
    plt.figure(figsize=(10,10))
    plt.subplot(1,1,1)
    plt.imshow(x_test[i],cmap='Greys')
    plt.title('real - '+str(false_test_positions.loc[i]['real'])+'\n'+'pred - '+str(false_test_positions.loc[i]['pred']))
    plt.show()