import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

loss_func = tf.keras.losses.MeanSquaredError()
loss_func_1 = tf.keras.losses.MeanAbsoluteError()

def loss_func(y_true , y_pred):
    return tf.reduce_mean((0.5)*((y_pred - y_true)**2))   # mean square error
  
x = tf.linspace(-2, 2 , 3000).numpy()
y = (x**2)
x_1 = tf.linspace(-2, 2 , 3000).numpy()
y1 = (2*x)
y_true = y + y1

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, shuffle = True)
x1_train, x1_test , y1_train , y1_test = train_test_split(x , y1 , test_size = 0.2 , shuffle = True)

#Model 1: x^2
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_dim = 1 , activation="elu" , name="layer1"),
    tf.keras.layers.Dense(5, activation = 'sigmoid' , name="layer2"),
    tf.keras.layers.Dense(1, name="layer3"),
])

#Model 2: 2x
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1 , input_dim = 1 , activation = "relu" , name="layer1"),
    tf.keras.layers.Dense(1,  name = "layer3")
])

model_1.compile(
    optimizer = 'Adam',
    loss = loss_func,
    metrics = ['accuracy']
)

model_2.compile(
    optimizer = 'Adam',
    loss = loss_func_1,
    metrics = ['accuracy']
)

train_data = model_1.fit(x_train , y_train, epochs =1000)
time.sleep(2)
print("Model 2: training")
train_data_1 = model_2.fit(x1_train , y1_train, epochs = 1000)

################################################################################################

# For x^2 + 2x

x_train , x_test , y_true_train , y_true_test = train_test_split(x , y_true , test_size = 0.2 , shuffle = True)
input_shape = tf.keras.layers.Input(shape=(1,))
full_model = tf.keras.models.Model(inputs = input_shape , outputs = tf.keras.layers.add([model_1(input_shape) , model_2(input_shape)]))

full_model.compile(
    optimizer = 'adam',
    loss = loss_func,
    metrics = ['accuracy']
)

full_model.fit(x_train , y_true_train , epochs = 50)
model_1.save("/Users/levanfuentes-parra/Desktop/Tai/Lab-Research/models/full_model.h5")

plt.plot(x, full_model(x) , label = "Predict Model: ")
plt.plot(x , y_true , label = "Real Model")
plt.legend()

