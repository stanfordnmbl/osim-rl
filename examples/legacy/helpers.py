from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate

def policy_nn(shape_in, shape_out, hidden_layers = 3, hidden_size = 32):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + (shape_in, )))
    for i in range(hidden_layers):
        actor.add(Dense(hidden_size))
        actor.add(Activation('relu'))
    actor.add(Dense(shape_out))
    actor.add(Activation('sigmoid'))
    return actor

def q_nn(nb_obs, nb_act, hidden_layers = 3, hidden_size = 64):
    action_input = Input(shape=(nb_act, ), name='action_input')
    observation_input = Input(shape=(1,) + (nb_obs, ), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    for i in range(hidden_layers):
        x = Dense(hidden_size)(x)
        x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return critic, action_input