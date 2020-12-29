#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
import numpy as np
from bd1_train.ppo import PPO
from bd1_train.srv import SaveAgent, SaveAgentResponse
from bd1_gazebo_env_interface.srv import Step, Reset, Configure
import matplotlib.pyplot as plt
import yaml

class StandUpTrain(object):        
    
    def __init__(self):
        
        rospy.init_node('universal_train_ppo')
        self.name = rospy.get_name()        
                        
        # MUST HAVE PARAMS
        self.env_interface_node_name = rospy.get_param("~env_interface_node_name", "")         
        self.req_actions = rospy.get_param("~actions", [])
        self.req_state = rospy.get_param("~state", [])   
        self.reward_type = rospy.get_param("~reward", "")   
        
        self.sub_method = rospy.get_param("~sub_method", 'clip') 
        if self.sub_method != 'clip' or self.sub_method != 'penalty':        
            self.sub_method = 'clip'
        
        self.step_duration = rospy.get_param("~step_duration_sec", 0.2)                
        self.num_episodes = rospy.get_param("~num_episodes", 1000)
        self.test_episodes = rospy.get_param("~test_episodes", 10)
        self.max_steps = rospy.get_param("~max_steps", 200)
        
        self.backup_saves = rospy.get_param("~backup_saves", 250)                        
        # HYPERPARAMETERS
        self.hyper_parameters = {}
        self.hyper_parameters['LR_A'] = rospy.get_param('~actor_learning_rate', 0.0001)
        self.hyper_parameters['LR_C'] = rospy.get_param('~critic_learning_rate', 0.0002)
        self.hyper_parameters['GAMMA'] = rospy.get_param('~gamma_reward_discount', 0.9)
        
        self.hyper_parameters['BATCH_SIZE'] = rospy.get_param('~batch_size', 32)
        self.hyper_parameters['ACTOR_UPDATE_STEPS'] = rospy.get_param('~actor_update_steps', 10)
        self.hyper_parameters['CRITIC_UPDATE_STEPS'] = rospy.get_param('~critic_update_steps', 10)
        
        self.hyper_parameters['KL_TARGET'] = rospy.get_param('~ppo_penalty_param1', 0.01)
        self.hyper_parameters['LAM'] = rospy.get_param('~ppo_penalty_param2', 0.5)
        self.hyper_parameters['EPSILON'] = rospy.get_param('~ppo_clip_param', 0.2)
        
        self.hyper_parameters['CRITIC_LAYER1_SIZE'] = rospy.get_param('~critic_layer1_size', 64)
        self.hyper_parameters['CRITIC_LAYER2_SIZE'] = rospy.get_param('~critic_layer2_size', 64)
        self.hyper_parameters['CRITIC_LAYER3_SIZE'] = rospy.get_param('~critic_layer3_size', 64)
        self.hyper_parameters['ACTOR_LAYER1_SIZE'] = rospy.get_param('~actor_layer1_size', 64)
        self.hyper_parameters['ACTOR_LAYER2_SIZE'] = rospy.get_param('~actor_layer2_size', 64)
        self.hyper_parameters['ACTOR_LAYER3_SIZE'] = rospy.get_param('~actor_layer3_size', 64)
        
        rospy.wait_for_service(self.env_interface_node_name+'/configure')
        self.env_config_srv = rospy.ServiceProxy(self.env_interface_node_name+'/configure', Configure)
        
        config = self.env_config_srv(self.req_state, self.req_actions, self.reward_type)        
        if not config.configured:
            rospy.logerr("[{}] ERROR! Configure environment interface failed! Exit.".format(self.name))
            exit()
        self.action_dim = config.actions_dim
        self.action_range = 1#config.action_range
        self.state_dim = config.state_dim
        self.servo_control = config.servo_control
        self.action_real_lim = config.action_real_lim
        
        rospy.loginfo("[{}] initializing PPO...".format(self.name))
                            
        self.agent = PPO(self.state_dim, self.action_dim, self.action_range, self.hyper_parameters, self.sub_method)
        self.mode = rospy.get_param("~mode",'train')
        self.plot_reward = True
        
        rospy.logwarn("[{}] PPO inited!".format(self.name))
        
        self.backup_num = rospy.get_param('~episode_backup_num', 500)
        self.save_path = rospy.get_param('~save_path', "/tmp")
        self.load_path = rospy.get_param('~load_path', None)
        if self.load_path is not None:
            self.agent.load(self.load_path)
                
        self.cmap = plt.get_cmap("tab10")
        
        # services init
        rospy.wait_for_service(self.env_interface_node_name+'/reset')
        self.env_reset_srv = rospy.ServiceProxy(self.env_interface_node_name+'/reset', Reset)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
        
        rospy.wait_for_service(self.env_interface_node_name+'/step')
        self.env_step_srv = rospy.ServiceProxy(self.env_interface_node_name+'/step', Step)
        rospy.loginfo("[{}] step service ready!".format(self.name))
                        
        rospy.Service("~change_train_test_mode", Empty, self.change_mode_cb)
        rospy.Service("~save_agent", SaveAgent, self.save_agent_cb)
        rospy.Service("~on_off_reward_plot", Empty, self.on_off_reward_plot_cb)                
                
    def on_off_reward_plot_cb(self, req):
        self.plot_reward = not self.plot_reward
        return []
        
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
                            
    def export_params(self, name):
        params = {}
        params["method"] = "PPO"
        params["step_duration_sec"] = self.step_duration
        
        params["num_episodes"] = self.num_episodes
        params["max_steps"] = self.max_steps
        params["sub_method"] = self.sub_method
        params["action_dim"] = self.action_dim
        params["action_range"] = self.action_range
        params["state_dim"] = self.state_dim
        
        params["env_interface_node_name"] = self.env_interface_node_name
        params["req_actions"] = self.req_actions
        params["req_state"] = self.req_state
        params["reward_type"] = self.reward_type
                        
        # hyper
        params["actor_learning_rate"] = self.hyper_parameters['LR_A']
        params["critic_learning_rate"] = self.hyper_parameters['LR_C']
        params["gamma_reward_discount"] = self.hyper_parameters['GAMMA']
        params["batch_size"] = self.hyper_parameters['BATCH_SIZE']
        params["actor_update_steps"] = self.hyper_parameters['ACTOR_UPDATE_STEPS']
        params["critic_update_steps"] = self.hyper_parameters['CRITIC_UPDATE_STEPS']
        params["ppo_penalty_param1"] = self.hyper_parameters['KL_TARGET']
        params["ppo_penalty_param2"] = self.hyper_parameters['LAM']
        params["ppo_clip_param"] = self.hyper_parameters['EPSILON']        
        params["critic_layer1_size"] = self.hyper_parameters['CRITIC_LAYER1_SIZE']
        params["critic_layer2_size"] = self.hyper_parameters['CRITIC_LAYER2_SIZE']
        params["critic_layer3_size"] = self.hyper_parameters['CRITIC_LAYER3_SIZE']
        params["actor_layer1_size"] = self.hyper_parameters['ACTOR_LAYER1_SIZE']
        params["actor_layer2_size"] = self.hyper_parameters['ACTOR_LAYER2_SIZE']
        params["actor_layer3_size"] = self.hyper_parameters['ACTOR_LAYER3_SIZE']        
        
        params["pretrained model"] = self.load_path                
        
        params["servo_control"] = self.servo_control
        params["action_real_lim"] = self.action_real_lim
        
        with open(self.save_path + "/"+name+"/config.yaml", 'w') as file:
            documents = yaml.dump(params, file)
        
    def save_all(self, name):
        self.save_agent(name)
        self.export_params(name)
        plt.savefig(self.save_path+"/"+name+"/rewards.png")
        
    def run(self):        
        if self.mode == 'train':
            all_episode_reward = []
            true_all_episode_reward = []
            for episode in range(self.num_episodes):
                state = np.array(self.env_reset_srv().state)
                episode_reward = 0
                for step in range(self.max_steps):
                    #print(state.shape, self.state_dim)
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
                true_all_episode_reward.append(episode_reward)
                
                plt.cla()                
                plt.plot(all_episode_reward, '-', color = self.cmap(0), label="episode reward discounted")                
                plt.legend()
                plt.title("Reward per episode")
                if self.plot_reward:                    
                    plt.pause(0.001)
                    
                if episode != 0 and episode % self.backup_saves == 0:
                    name = "backup-{}".format(episode)
                    self.save_all(name)                            
            self.save_all("fully_trained")                
            rospy.logwarn("[{}] Training is complete!".format(self.name))
                    
        for episode in range(self.test_episodes):
            state = np.array(self.env_reset_srv().state)
            episode_reward = 0
            for step in range(self.max_steps):                
                action = self.agent.get_action(state, greedy=True)
                srd = self.env_step_srv(self.step_duration, action.tolist())
                state = np.array(srd.state)
                episode_reward += srd.reward
                print(srd.reward)
                if srd.done:
                    break
            rospy.loginfo("[{}] Teting | Episode: {}/{} | Episode Reward: {:.4f}".format(self.name, episode, self.test_episodes,  episode_reward))            
            
                    
    
if __name__ == '__main__' :
    sut = StandUpTrain()
    sut.run()
    
