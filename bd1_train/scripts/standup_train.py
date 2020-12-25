#!/usr/bin/env python
# coding: utf-8

import rospy
from bd1_environment_interface.srv import SetAction, SetVectAction, GetStateAndReward, GetVectStateAndReward
from std_srvs.srv import Empty

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow.keras.activations import sigmoid

from bd1_train.ddpg import DDPG
import os
from bd1_train.srv import SaveAgent, SaveAgentResponse
import matplotlib.pyplot as plt

class StandUpTrain(object):        
    
    def __init__(self):
        self.name = "standup_train"
        rospy.init_node(self.name)
        
        # HYPERPARAMETERS
        self.episode_duration = rospy.Duration(rospy.get_param("~episode_duration_sec", 0.2))
                
        self.num_episodes = rospy.get_param("~num_episodes", 10000)
        self.max_steps = rospy.get_param("~max_steps", 1000)                        
        
        self.hyper_parameters = {}
        self.hyper_parameters['LR_A'] = rospy.get_param('~actor_learning_rate', 0.001)
        self.hyper_parameters['LR_C'] = rospy.get_param('~critic_learning_rate', 0.002)
        self.hyper_parameters['GAMMA'] = rospy.get_param('~gamma_reward_discount', 0.9)
        self.hyper_parameters['TAU'] = rospy.get_param('~tau_soft_replacement', 0.01)
        self.hyper_parameters['MEMORY_CAPACITY'] = rospy.get_param('~memcap_replase_buffer_size', 10000)
        self.hyper_parameters['VAR'] = rospy.get_param('~var_control_extrapolation', 2)
        self.hyper_parameters['BATCH_SIZE'] = rospy.get_param('~batch_size', 32)
                        
        rospy.loginfo("[{}] initializing DDPG...".format(self.name))
                            
        # inner values
        self.step = 0
        self.episode = 0
        self.state = None
        self.all_episode_reward = []
        self.max_episode_reward = -100
        self.all_max_episode_reward = []
        self.all_episode_mean_reward = []
        self.all_steps_reward  = []
        self.episode_reward = 0
        self.action = None
        self.train_test_mode = 'train'
        self.switch_mode = False
        self.continue_training = False
                
        #self.agent = DDPG(8, 14, 1, self.hyper_parameters)
        self.agent = DDPG(3, 9, 1, self.hyper_parameters)                
        
        rospy.logwarn("[{}] DDPG inited!".format(self.name))
        
        self.save_path = rospy.get_param('~save_path', "/tmp")
        self.load_path = rospy.get_param('~load_path', None)
        if self.load_path is not None:
            self.agent.load(self.load_path)
        
        self.update_plot = False
        self.reloading = False        
        self.cmap = plt.get_cmap("tab10")
        
        # services init
        rospy.wait_for_service('environment_interface_standup/reset')
        self.reset_srv = rospy.ServiceProxy('environment_interface_standup/reset', Empty)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
                
        rospy.wait_for_service('environment_interface_standup/get_vect_state_and_reward')
        self.get_state_and_reward_srv = rospy.ServiceProxy('environment_interface_standup/get_vect_state_and_reward', GetVectStateAndReward)
        rospy.loginfo("[{}] state and reward service ready!".format(self.name))
        
        #rospy.wait_for_service('environment_interface_standup/set_action')
        #self.set_action_srv = rospy.ServiceProxy('environment_interface_standup/set_action', SetAction)
        
        rospy.wait_for_service('environment_interface_standup/set_vect_action')
        self.set_action_srv = rospy.ServiceProxy('environment_interface_standup/set_vect_action', SetVectAction)
        rospy.loginfo("[{}] set vect action service ready!".format(self.name))
        
        
        rospy.wait_for_service('gazebo/pause_physics')
        self.pause_srv = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        
        rospy.wait_for_service('gazebo/unpause_physics')
        self.unpause_srv = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        
        rospy.Service("~change_train_test_mode", Empty, self.change_mode_cb)
        rospy.Service("~save_agent", SaveAgent, self.save_agent_cb)                
        
        rospy.Timer(self.episode_duration, self.train_cb)
        
    def save_agent_cb(self, req):         
        return SaveAgentResponse(self.save_agent(req.name))
    
    def save_agent(self, name):
        dir_path = self.save_path + '/'+ name
        self.agent.save(dir_path)
        rospy.logwarn("[{}] agent saved to {}".format(self.name, dir_path))
        return dir_path
        
        
    def change_mode_cb(self, req):
        self.switch_mode = True
        #if self.train_test_mode == 'train':
            #self.train_test_mode = 'test'
        #else:
            #self.train_test_mode = 'train'
        return []
        
        
    def vectorize_state(self, sar):
        
        #state = [sar.up_p_r,
                 #sar.mid_p_r,
                 #sar.feet_p_r,
                 #sar.up_p_l,
                 #sar.mid_p_l,
                 #sar.feet_p_l,
                 #sar.neck_p,
                 #sar.head_p,
                 #sar.pose_x, sar.pose_y, sar.pose_z,
                 #sar.rot_r, sar.rot_p, sar.rot_y]        
        #print(sar)
        return np.array(sar[:6]+sar[8:11])           
    
    def start_new_episode(self):
        rospy.logwarn("[{}] Starting new episode!".format(self.name))          
        
        if( self.agent.pointer < self.hyper_parameters['MEMORY_CAPACITY']):
            rospy.loginfo("[{}] {}\{} for learning.".format(self.name,self.agent.pointer, self.hyper_parameters['MEMORY_CAPACITY']))
        
        if self.episode == 0:
            self.all_episode_reward.append(self.episode_reward)            
        else:            
            self.all_episode_reward.append(self.all_episode_reward[-1] * 0.9 + self.episode_reward * 0.1)
        self.all_episode_mean_reward.append(self.episode_reward/self.step)
        self.episode+=1
        self.reset_srv()        
        
        self.step = 0
        self.episode_reward = 0
        init_sar = self.get_state_and_reward_srv()
        self.state = self.vectorize_state(init_sar.state)
        self.action = self.agent.get_action(self.state)
        self.set_action_srv(self.action.tolist() + [0]*5) 
        self.all_max_episode_reward.append(self.max_episode_reward)
        self.max_episode_reward = -100
        self.prev_all_step_reward = self.all_steps_reward
        self.all_steps_reward = []
        self.update_plot = True
        self.reloading = False
        
        
    
    def train_cb(self, event):    
        if self.episode >= self.num_episodes:
            self.save_agent("fully_trained")
        
        if self.reloading:
            return
        if self.train_test_mode == "train":            
            
            
            if self.episode == 0 and self.step == 0:
                rospy.logwarn("[{}] Initial reset of environment...".format(self.name))
                self.reset_srv()
                init_sar = self.get_state_and_reward_srv()
                self.state = self.vectorize_state(init_sar.state)
                self.episode_reward = 0
                self.action = self.agent.get_action(self.state)
                self.set_action_srv(self.action.tolist() + [0]*5) 
                self.step+=1
                return # maybe
                
            rospy.loginfo("[{}] [Train] Episode: {}/{}, Step: {}/{}, Episode reward: {:.4f}".format(self.name, self.episode, self.num_episodes, self.step, self.max_steps, self.episode_reward))
                
            self.step+=1
            
            new_sar = self.get_state_and_reward_srv()
            state_ = self.vectorize_state(new_sar.state)
            reward = new_sar.reward
            done = new_sar.episode_end        
            if done:
                reward -= 0.01
            self.all_steps_reward.append(reward)
            self.agent.store_transition(self.state, self.action, reward, state_) # we store here PREVIOUS action
            
            #print("{} -> {}".format(self.state, state_))
            
            self.state = state_
            
            self.action = self.agent.get_action(self.state)        
            self.set_action_srv(self.action.tolist() + [0]*5) 
                                            
            if self.agent.pointer > self.hyper_parameters['MEMORY_CAPACITY'] :    
                self.pause_srv()
                self.agent.learn() # NOTE if it long operation maybe pause simulation?            
                self.unpause_srv()
                    
            if reward > self.max_episode_reward:
                self.max_episode_reward = reward
            self.episode_reward += reward
            
            # start new episode!
            if( done or self.step >= self.max_steps ):  
                if self.switch_mode:
                    self.train_test_mode = "test"
                    self.switch_mode = False
                    rospy.logwarn("[{}] Switch mode to TEST".format(self.name))   
                    self.reset_srv()
                    init_sar = self.get_state_and_reward_srv()
                    self.state = self.vectorize_state(init_sar.state)
                    #self.all_episode_reward.append(self.episode_reward)
                    self.episode_reward = 0
                    self.action = self.agent.get_action(self.state, greedy = True).numpy()
                    self.set_action_srv(self.action.tolist() + [0]*5) 
                    
                    return
                self.reloading = True
                self.start_new_episode()
                
                
        if self.train_test_mode == "test":
            if self.switch_mode:
                self.train_test_mode = "train"
                rospy.logwarn("[{}] Switch mode to TRAIN".format(self.name))            
                self.switch_mode = False
                self.start_new_episode()
                return
            
            self.step+=1
            
            sar = self.get_state_and_reward_srv()
            self.state = self.vectorize_state(sar.state)
            self.episode_reward += sar.reward
            self.action = self.agent.get_action(self.state, greedy = True).numpy()
            self.set_action_srv(self.action.tolist() + [0]*5) 
            
            rospy.loginfo("[{}] [Test] Step {}/{}, Episode reward: {:.4f}".format(self.name, self.step, self.max_steps, self.episode_reward))
            
            if( sar.episode_end or self.step >= self.max_steps):
                rospy.logwarn("[{}] Starting new episode!".format(self.name))            
                self.reset_srv()                
                self.step = 0
                self.episode_reward = 0
                init_sar = self.get_state_and_reward_srv()
                self.state = self.vectorize_state(init_sar.state)
                self.action = self.agent.get_action(self.state)
                self.set_action_srv(self.action.tolist() + [0]*5) 
                                                    
                            
        
    def run(self):
        #rospy.spin()
        while(not rospy.is_shutdown()):         
            if self.update_plot:            
                #plt.cla()
                #plt.plot(self.all_episode_reward, '.', label="sum reward")
                xticks = [self.episode + x * (1/(len(self.prev_all_step_reward)+1) ) for x in range(len(self.prev_all_step_reward))]
                plt.plot(xticks, self.prev_all_step_reward, '-', color = self.cmap(0), label="steps_reward")
                plt.plot(self.episode, self.all_episode_reward[-1], '.', color = self.cmap(1), label="steps_reward")
                #plt.plot(self.all_max_episode_reward, '.', label="max_reward")
                #plt.plot(self.all_episode_mean_reward, '.', label="mean_reward")
                plt.title('Episode Reward')
                #plt.legend()
                plt.pause(0.1)
                self.update_plot = False
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
