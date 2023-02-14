import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Custom Loss Function
# def new_loss_func(atual_v , predict_v):
    
x = tf.linspace(-2, 2 , 1000).numpy()    #Generate Data from -2 to 2 with 1000 data
y = (x**2)+2                             # y = x^2 + 2
# split test data and train data with 80% of train, 20% test
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , shuffle = True)

#Generate Model
model = tf.keras.models.Sequential([                   
    tf.keras.layers.Dense(1, input_dim = 1 , activation="elu" , name="layer1"),
    tf.keras.layers.Dense(5, activation = 'sigmoid' , name="layer2"),
    tf.keras.layers.Dense(1, name="layer3"),
    
])

model.compile(
    optimizer = 'Adam',
    loss = 'mean_squared_error',
    metrics = ['accuracy']
)
# Plot data
train_data = model.fit(x_train , y_train, epochs =800 , shuffle = True)
plt.plot(x , model(x) , label = 'predictions')
plt.plot(x,y , label = 'Real Data')
plt.legend()
