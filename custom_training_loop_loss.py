import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
optimizer = tf.keras.optimizers.Adam(
    learning_rate= 0.001,
)
def loss_func(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true , y_pred)
    # or 
    # mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse
def gradient_step(model , x , y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_func(y, y_pred)
    return loss_value , tape.gradient(loss_value , model.trainable_variables)
def train_loop(model, train_dataset, epochs):
    train_loss = []
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.Accuracy()
    for i in range(epochs):
        for (x_batch, y_batch) in train_dataset:
            loss_value , gradient = gradient_step(model , x_batch , y_batch)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            loss_avg.update_state(loss_value)
            accuracy.update_state(y_batch , model(x_batch , training = True))
        train_loss.append(loss_avg.result().numpy())
        print("Epoch: {}  Loss: {} Accutacy: {}".format(i+1, loss_avg.result() , accuracy.result()))
    return train_loss
def main():
    x = tf.linspace(-2.0 , 2.0 , 300)
    y = x**2
    y1 = 2*x
    y_true = y + y1
    len_train = int(0.8*len(x))
    x_train , y_true_train = x[:len_train] , y_true[:len_train]
    x_test , y_true_test = x[len_train:] , y_true[len_train:]
    # x_train, x_test, y_true_train, y_true_test = train_test_split(x, y_true, test_size=0.2, shuffle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_true_train)).shuffle(buffer_size = len(x_train)).batch(32)
    input_shape = tf.keras.layers.Input(sshape=(1,))
    model_1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_dim=1, activation="elu", name="layer1"),
        tf.keras.layers.Dense(5, activation='sigmoid', name="layer2"),
        tf.keras.layers.Dense(1, name="layer3"),
    ])
    model_2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_dim=1, activation="relu", name="layer1"),
        tf.keras.layers.Dense(1, name="layer3")
    ])
    full_model = tf.keras.models.Model(inputs=input_shape, outputs=tf.keras.layers.add([model_1(input_shape), model_2(input_shape)]))

    loss_results = train_loop(model=full_model, train_dataset=train_dataset, epochs=200)
    plt.plot(x, full_model(x), label="Predict Model")
    plt.plot(x, y_true, label="Real Model")
    plt.legend()

if __name__ == "__main__":
    main()
