import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A3C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, layers=[]):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.build_critic(inp_dim, lr, layers=layers)

    def build_critic(self, env_dim, lr, layers=[]):
        state_input = Input(shape=(env_dim,))

        for layer in layers:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            is_input = layer['is_input'] if 'is_input' in layer else None
            is_output = layer['is_output'] if 'is_output' in layer else None

            if layer['type'] == 'dense':
                if is_input: x = Dense(neurons, activation=activation, name='critic_input')(state_input)
                elif is_output: out_value = Dense(1, activation=activation, name='critic_output')(x)
                else: x = Dense(neurons, activation=activation)(x)
            if layer['type'] == 'lstm':
                if is_input: x = LSTM(neurons, activation=activation, return_sequences=return_seq, name='critic_input')(state_input)
                elif is_output: out_value = LSTM(1, activation=activation, return_sequences=return_seq, name='critic_output')(x)
                else: x = LSTM(neurons, return_sequences=return_seq)(x)
            if layer['type'] == 'dropout':
                x = Dropout(dropout_rate)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        model.summary()

        return model

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
