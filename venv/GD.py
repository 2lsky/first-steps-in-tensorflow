import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot  as plt
import seaborn as sns
tf.random.set_seed(42)
t_mass=[0.0+0.005*i for i in range(0,100)]
t=tf.constant(t_mass,shape=[len(t_mass)])
noise=tf.random.normal(shape=[100],stddev=0.001)
a_true=1
w_true=2*3.14/0.6
y_true=a_true*tf.sin(w_true*t)
EPOCHES=1000
w=tf.Variable(1.0)
a=tf.Variable(1.0)
opt=tf.optimizers.Adam(learning_rate=0.1)
for i in range(EPOCHES):
    with tf.GradientTape() as tape:
        f=a*tf.sin(w*t)
        loss=tf.reduce_mean(tf.square(y_true-f))
    da,dw=tape.gradient(loss,[a,w])
    opt.apply_gradients(zip([da,dw],[a,w]))
print(a,w)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
sns.scatterplot(x=t,y=y_true)
sns.lineplot(x=t,y=a*tf.sin(w*t))

plt.show()