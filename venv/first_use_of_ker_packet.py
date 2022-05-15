import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
c=np.array([40,20,30]).reshape(-1,1)
f=np.array([40,20,30]).reshape(-1,1)
test=np.array([-200]).reshape(-1,1)
model=keras.Sequential()
model.add(Dense(units=1,input_shape=(-1,1),activation='linear'))
model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(0.1))
history=model.fit(c,f,epochs=500,verbose=0)
plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.plot(history.history['loss'])
plt.show()
print(model.get_weights())
print(model.predict(test))