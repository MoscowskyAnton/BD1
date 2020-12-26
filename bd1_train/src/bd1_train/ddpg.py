#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os

# taken from 
# https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_DDPG.py


class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, action_dim, state_dim, action_range, hyper_params):
        self.hyper_params = hyper_params
        self.memory = np.zeros((self.hyper_params['MEMORY_CAPACITY'], state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = self.hyper_params['VAR']

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.hyper_params['TAU'])  # soft replacement

        self.actor_opt = tf.optimizers.Adam(self.hyper_params['LR_A'])
        self.critic_opt = tf.optimizers.Adam(self.hyper_params['LR_C'])

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.var), -self.action_range, self.action_range
        )  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(self.hyper_params['MEMORY_CAPACITY'], size=self.hyper_params['BATCH_SIZE'])
        datas = self.memory[indices, :]
        states = datas[:, :self.state_dim]
        actions = datas[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = datas[:, -self.state_dim - 1:-self.state_dim]
        states_ = datas[:, -self.state_dim:]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            y = rewards + self.hyper_params['GAMMA'] * q_
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape2: #NOTE https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f says taht TF recommends use different tapes for different networks
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape2.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.hyper_params['MEMORY_CAPACITY']  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self, path):
        """
        save trained weights
        :return: None
        """
        #path = os.path.join('model', '_'.join(['DDPG', 'BD1_standup']))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self, path):
        """
        load trained weights
        :return: None
        """
        #path = os.path.join('model', '_'.join(['DDPHG', 'BD1_standup']))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)
