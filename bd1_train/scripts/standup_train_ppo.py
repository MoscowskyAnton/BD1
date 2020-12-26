#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
import numpy as np
from bd1_train.ppo import PPO
from bd1_train.srv import SaveAgent, SaveAgentResponse
from bd1_gazebo_env_interface.srv import Step, Reset
import matplotlib.pyplot as plt

class StandUpTrain(object):        
    
    def __init__(self):
        
        rospy.init_node('ppo_train_standup')
        self.name = rospy.get_name()
        # HYPERPARAMETERS
        self.step_duration = rospy.get_param("~step_duration_sec", 0.2)
                
        self.num_episodes = rospy.get_param("~num_episodes", 1000)
        self.max_steps = rospy.get_param("~max_steps", 200)                        
        
        self.hyper_parameters = {}
        self.hyper_parameters['LR_A'] = rospy.get_param('~actor_learning_rate', 0.0001)
        self.hyper_parameters['LR_C'] = rospy.get_param('~critic_learning_rate', 0.0002)
        self.hyper_parameters['GAMMA'] = rospy.get_param('~gamma_reward_discount', 0.9)
        #self.hyper_parameters['TAU'] = rospy.get_param('~tau_soft_replacement', 0.01)
        #self.hyper_parameters['MEMORY_CAPACITY'] = rospy.get_param('~memcap_replase_buffer_size', 10000)
        #self.hyper_parameters['VAR'] = rospy.get_param('~var_control_extrapolation', 2)
        self.hyper_parameters['BATCH_SIZE'] = rospy.get_param('~batch_size', 32)
        self.hyper_parameters['ACTOR_UPDATE_STEPS'] = rospy.get_param('~actor_update_steps', 10)
        self.hyper_parameters['CRITIC_UPDATE_STEPS'] = rospy.get_param('~critic_update_steps', 10)
        
        self.hyper_parameters['KL_TARGET'] = rospy.get_param('~ppo_penalty_param1', 0.01)
        self.hyper_parameters['LAM'] = rospy.get_param('~ppo_penalty_param2', 0.5)
        self.hyper_parameters['EPSILON'] = rospy.get_param('~ppo_clip_param', 0.2)
                        
        rospy.loginfo("[{}] initializing PPO...".format(self.name))
                            
        self.agent = PPO(12, 3, 1, self.hyper_parameters, 'penalty')
        self.mode = 'train'
        
        rospy.logwarn("[{}] PPO inited!".format(self.name))
        
        self.backup_num = rospy.get_param('~episode_backup_num', 500)
        self.save_path = rospy.get_param('~save_path', "/tmp")
        self.load_path = rospy.get_param('~load_path', None)
        if self.load_path is not None:
            self.agent.load(self.load_path)
                
        self.cmap = plt.get_cmap("tab10")
        
        # services init
        rospy.wait_for_service('simple_standup_interface/reset')
        self.env_reset_srv = rospy.ServiceProxy('simple_standup_interface/reset', Reset)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
        
        rospy.wait_for_service('simple_standup_interface/step')
        self.env_step_srv = rospy.ServiceProxy('simple_standup_interface/step', Step)
        rospy.loginfo("[{}] step service ready!".format(self.name))
                        
        rospy.Service("~change_train_test_mode", Empty, self.change_mode_cb)
        rospy.Service("~save_agent", SaveAgent, self.save_agent_cb)                
                
        
    def save_agent_cb(self, req):         
        return SaveAgentResponse(self.save_agent(req.name))
    
    def save_agent(self, name):
        dir_path = self.save_path + '/'+ name
        self.agent.save(dir_path)
        rospy.logwarn("[{}] agent saved to {}".format(self.name, dir_path))
        return dir_path
                
    def change_mode_cb(self, req):
        self.mode = 'test'
        return []                                                                        
                            
        
    def run(self):        
        #while(not rospy.is_shutdown()):       
        all_episode_reward = []
        for episode in range(self.num_episodes):
            state = np.array(self.env_reset_srv().state)
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.agent.get_action(state)
                srd = self.env_step_srv(self.step_duration, action.tolist())
                state_ = np.array(srd.state)
                self.agent.store_transition(state, action, srd.reward)
                state = state_
                episode_reward += srd.reward
                
                if len(self.agent.state_buffer) > self.hyper_parameters["BATCH_SIZE"]:
                    self.agent.finish_path(state_, srd.done)
                    self.agent.update()
                if srd.done:
                    break
            self.agent.finish_path(state_, srd.done)
            rospy.loginfo("[{}] Training | Episode: {}/{} | Episode Reward: {:.4f}".format(self.name, episode+1, self.num_episodes, episode_reward))                        
            
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
                
            plt.cla()
            plt.plot(all_episode_reward, '-', color = self.cmap(0), label="episode_rewards")
            plt.title("Rewards per episode (with discount)")
            plt.pause(0.001)
        self.save_agent("fully_trained")
        plt.savefig(self.save_path+"/fully_trained/rewards.png")
            
        epi_num = 1
        while(not rospy.is_shutdown()):
            state = np.array(self.env_reset_srv().state)
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.agent.get_action(state, greedy=True)
                srd = self.env_step_srv(self.step_duration, action.tolist())
                episode_reward += srd.reward
                if srd.done:
                    break
            rospy.loginfo("[{}] Teting | Episode: {}/inf | Episode Reward: {:.4f}".format(self.name, epi_num,  episode_reward))
            epi_num+=1
            
                    
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
