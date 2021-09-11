import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        value, logits = self.predict_on_batch(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value