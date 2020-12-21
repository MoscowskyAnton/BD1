#!/usr/bin/env python
# coding: utf-8

import rospy
from bd1_environment_interface.srv import SetAction, GetStateAndReward
from std_srvs.srv import Empty

import tensorflow as tf
import tensorlayer as tl

# DQN 
# example is https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_DQN.py
 

#def to_one_hot(i, n_classes=None):
    #a = np.zeros(n_classes, 'uint8')
    #a[i] = 1
    #return a

## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.

def get_model(inputs_shape, output_shape):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(output_shape, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")

class StandUpTrain(object):        
    
    def __init__(self):
        self.name = "standup_train"
        rospy.init_node(self.name)
        
        # HYPERPARAMETERS
        self.episode_duration = rospy.Duration(rospy.get_param("~episode_duration_sec", 0.2))
        self.alg_name = "DQN"
        self.lambd = rospy.get_param("~lambda_decay_factor", 0.99)
        self.e = rospy.get_param("~e_greedy_extrapolation", 0.1)
        self.num_episodes = rospy.get_param("~num_episodes", 10000)
        self.max_steps = rospy.get_param("~max_steps", 10000)
        self.learning_rate = rospy.get_param("~learning_rate", 0.1)
        
        rospy.loginfo("[{}] initializing DQN...".format(self.name))
        # DQN
        self.qnetwork = get_model([None, 22], 16)
        self.qnetwork.train()
        self.train_weights = self.qnetwork.trainable_weights
        self.optimizer = tf.optimizers.SGD(learning_rate = self.learning_rate)
        
        
        # services init
        rospy.wait_for_service('environment_interface_standup/reset')
        self.set_model_state_srv = rospy.ServiceProxy('environment_interface_standup/reset', Empty)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
                
        rospy.wait_for_service('environment_interface_standup/get_state_and_reward')
        self.set_model_state_srv = rospy.ServiceProxy('environment_interface_standup/get_state_and_reward', GetStateAndReward)
        rospy.loginfo("[{}] state and reward service ready!".format(self.name))
        
        rospy.wait_for_service('environment_interface_standup/set_action')
        self.set_model_state_srv = rospy.ServiceProxy('environment_interface_standup/set_action', SetAction)
        rospy.loginfo("[{}] set action service ready!".format(self.name))
        
        rospy.Timer(self.episode_duration, self.train_cb)
        
    def train_cb(self, event):
        #rospy.logwarn("Tik-tok")
        pass
        
        
    def run(self):
        rospy.spin()
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
