#!/usr/bin/env python
# coding: utf-8

import rospy
from bd1_environment_interface.srv import SetAction, GetStateAndReward
from std_srvs.srv import Empty

import tensorflow as tf
import tensorlayer as tl
import numpy as np
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
        self.step = 0
        self.episode = 0
        
        rospy.logwarn("[{}] DQN inited!".format(self.name))
        
        # services init
        rospy.wait_for_service('environment_interface_standup/reset')
        self.reset_srv = rospy.ServiceProxy('environment_interface_standup/reset', Empty)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
                
        rospy.wait_for_service('environment_interface_standup/get_state_and_reward')
        self.get_state_and_reward_srv = rospy.ServiceProxy('environment_interface_standup/get_state_and_reward', GetStateAndReward)
        rospy.loginfo("[{}] state and reward service ready!".format(self.name))
        
        rospy.wait_for_service('environment_interface_standup/set_action')
        self.set_action_srv = rospy.ServiceProxy('environment_interface_standup/set_action', SetAction)
        rospy.loginfo("[{}] set action service ready!".format(self.name))
        
        rospy.Timer(self.episode_duration, self.train_cb)
        
    def train_cb(self, event):
        #rospy.logwarn("Tik-tok")
        
        if self.episode == 0 and self.step == 0:
            rospy.logwarn("[{}] Initial reset of environment...".format(self.name))
            self.reset_srv()
        
        if self.episode >= self.num_episodes:
            # finish training
            pass        
        
        # get state and reward
        sar_full = self.get_state_and_reward_srv()
        #print(sar_full)
        self.step+=1
        
        # reload episode
        if( sar_full.episode_end or self.step >= self.max_steps ):
            rospy.logwarn("[{}] Starting new episode!".format(self.name))
            # start new episode!
            self.reset_srv()
            self.episode+=1
            self.step = 0
            return
        
        rospy.loginfo("[{}] Episode: {}\{}, Step: {}\{}, Reward: {}".format(self.name, self.episode, self.num_episodes, self.step, self.max_steps, sar_full.reward))
        
        # vectorize state
        sar = sar_full.state
        #print(sar)
        state = [sar.up_p_r, sar.up_v_r,
                 sar.mid_p_r, sar.mid_v_r,
                 sar.feet_p_r, sar.feet_v_r,
                 sar.up_p_l, sar.up_v_l,
                 sar.mid_p_l, sar.mid_v_l,
                 sar.feet_p_l, sar.feet_v_l,
                 sar.neck_p, sar.neck_v,
                 sar.head_p, sar.head_v,
                 sar.pose_x, sar.pose_y, sar.pose_z,
                 sar.rot_r, sar.rot_p, sar.rot_y]
                
        
        # choose an action by greedily
        allQ = self.qnetwork(np.asarray([state], dtype=np.float32)).numpy()
        #rospy.loginfo(allQ)
        #a = np.argmax(allQ, 1)
        
    def run(self):
        rospy.spin()
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
