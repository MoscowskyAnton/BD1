#!/usr/bin/env python
# coding: utf-8

import rospy
from bd1_environment_interface.srv import SetAction, SetVectAction, GetStateAndReward
from std_srvs.srv import Empty

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow.keras.activations import sigmoid

from bd1_train.ddpg import DDPG

class StandUpTrain(object):        
    
    def __init__(self):
        self.name = "standup_train"
        rospy.init_node(self.name)
        
        # HYPERPARAMETERS
        self.episode_duration = rospy.Duration(rospy.get_param("~episode_duration_sec", 0.2))
                
        self.num_episodes = rospy.get_param("~num_episodes", 10000)
        self.max_steps = rospy.get_param("~max_steps", 10000)        
        
        self.step = 0
        self.episode = 0
        self.state = None
        self.all_episode_reward = []
        self.episode_reward = 0
        
        self.hyper_parameters = {}
        self.hyper_parameters['LR_A'] = rospy.get_param('~actor_learning_rate', 0.001)
        self.hyper_parameters['LR_C'] = rospy.get_param('~critic_learning_rate', 0.002)
        self.hyper_parameters['GAMMA'] = rospy.get_param('~gamma_reward_discount', 0.9)
        self.hyper_parameters['TAU'] = rospy.get_param('~tau_soft_replacement', 0.01)
        self.hyper_parameters['MEMORY_CAPACITY'] = rospy.get_param('~memcap_replase_buffer_size', 10000)
        self.hyper_parameters['VAR'] = rospy.get_param('~var_control_extrapolation', 2)
        self.hyper_parameters['BATCH_SIZE'] = rospy.get_param('~batch_size', 32)
                        
        rospy.loginfo("[{}] initializing DDPG...".format(self.name))
                
        self.agent = DDPG(16, 22, 1, self.hyper_parameters)                
        
        rospy.logwarn("[{}] DDPG inited!".format(self.name))
        
        # services init
        rospy.wait_for_service('environment_interface_standup/reset')
        self.reset_srv = rospy.ServiceProxy('environment_interface_standup/reset', Empty)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
                
        rospy.wait_for_service('environment_interface_standup/get_state_and_reward')
        self.get_state_and_reward_srv = rospy.ServiceProxy('environment_interface_standup/get_state_and_reward', GetStateAndReward)
        rospy.loginfo("[{}] state and reward service ready!".format(self.name))
        
        #rospy.wait_for_service('environment_interface_standup/set_action')
        #self.set_action_srv = rospy.ServiceProxy('environment_interface_standup/set_action', SetAction)
        
        rospy.wait_for_service('environment_interface_standup/set_vect_action')
        self.set_action_srv = rospy.ServiceProxy('environment_interface_standup/set_vect_action', SetVectAction)
        rospy.loginfo("[{}] set vect action service ready!".format(self.name))
        
        rospy.Timer(self.episode_duration, self.train_cb)
        
    def vectorize_state(self, sar):
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
        return np.array(state)                
    
    def train_cb(self, event):        
                        
        if self.episode == 0 and self.step == 0:
            rospy.logwarn("[{}] Initial reset of environment...".format(self.name))
            self.reset_srv()
            init_sar = self.get_state_and_reward_srv()
            self.state = self.vectorize_state(init_sar.state)
            self.episode_reward = 0
            
        rospy.loginfo("[{}] Episode: {}/{}, Step: {}/{}, Episode reward: {:.4f}".format(self.name, self.episode, self.num_episodes, self.step, self.max_steps, self.episode_reward))
            
        self.step+=1
            
        action = self.agent.get_action(self.state)
        
        self.set_action_srv(action.tolist()) # TODO check 
        
        new_sar = self.get_state_and_reward_srv()
        state_ = self.vectorize_state(new_sar.state)
        reward = new_sar.reward
        done = new_sar.episode_end
        
        self.agent.store_transition(self.state, action, reward, state_)
        
        if self.agent.pointer > self.hyper_parameters['MEMORY_CAPACITY'] :
            self.agent.learn() # NOTE if it long operation maybe pause simulation?
        
        self.state = state_
        self.episode_reward += reward
        
        # start new episode!
        if( done or self.step >= self.max_steps ):                        
            rospy.logwarn("[{}] Starting new episode!".format(self.name))            
            if self.episode == 0:
                self.all_episode_reward.append(self.episode_reward)
            else:
                self.all_episode_reward.append(self.all_episode_reward[-1] * 0.9 + self.episode_reward * 0.1)
            self.reset_srv()
            self.episode+=1
            self.step = 0
            self.episode_reward = 0
                            
        #if self.episode >= self.num_episodes:
            ## finish training
            #pass        
                
        #self.step+=1            
        
        #rospy.loginfo("[{}] Episode: {}\{}, Step: {}\{}, Reward: {}".format(self.name, self.episode, self.num_episodes, self.step, self.max_steps, init_sar.reward))

                
        ## shot throw net 
        #allQ = self.qnetwork(np.asarray([state], dtype=np.float32)).numpy()
        ##rospy.loginfo(allQ)
        #choosen_action = allQ[0].tolist() 
        
        ## exploration
        #if np.random.rand(1) < self.e:
            #self.set_action_srv(np.random.uniform(size=(16)).tolist())
        #else:
            ##send chosen action
            #self.set_action_srv(choosen_action)
         
        #updated_sar = self.get_state_and_reward_srv()
        #updated_state = vectorize_state(updated_sar.state)
        #updated_Q = self.qnetwork(np.asarray([updated_state], dtype=np.float32)).numpy()
        
        #targetQ = allQ
        ##targetQ 
        
        ## reload episode
        #if( first_sar.episode_end or self.step >= self.max_steps ):
            #rospy.logwarn("[{}] Starting new episode!".format(self.name))
            ## start new episode!
            #self.reset_srv()
            #self.episode+=1
            #self.step = 0
            #return
        #a = np.argmax(allQ, 1)
        
    def run(self):
        rospy.spin()
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
