import tensorflow as tf


class DNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, input):
        y = self(input)
        return tf.math.argmax(y, 1).numpy()

