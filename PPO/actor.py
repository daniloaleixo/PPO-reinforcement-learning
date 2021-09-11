import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import Adam
from .agent import Agent
from .ppo_loss import proximal_policy_optimization_loss

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, loss_clipping, entropy_loss, layers=[]):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.build_actor(inp_dim, out_dim, lr, loss_clipping, entropy_loss, layers=layers)


    def build_actor(self, env_dim, act_dim, lr, loss_clipping, entropy_loss, layers=[]):
        state_input = Input(shape=(env_dim,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(act_dim,))


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
                if is_input: x = Dense(neurons, activation=activation, name='actor_input')(state_input)
                elif is_output: out_actions = Dense(act_dim, activation=activation, name='actor_output')(x)
                else: x = Dense(neurons, activation=activation)(x)
            if layer['type'] == 'lstm':
                if is_input: x = LSTM(neurons, activation=activation, return_sequences=return_seq, name='actor_input')(state_input)
                elif is_output: out_actions = LSTM(act_dim, activation=activation, return_sequences=return_seq, name='actor_output')(x)
                else: x = LSTM(neurons, return_sequences=return_seq)(x)
            if layer['type'] == 'dropout':
                x = Dropout(dropout_rate)(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=lr),
                      loss=[
                          proximal_policy_optimization_loss(
                            advantage=advantage,
                            old_prediction=old_prediction,
                            loss_clipping=loss_clipping,
                            entropy_loss=entropy_loss
                          )
                        ]
                    )
        model.summary()

        return model

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
