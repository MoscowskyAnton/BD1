#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
import numpy as np
import gym
from gym import spaces
from bd1_gazebo_env_interface.srv import Step, Reset, Configure

class BD1GazeboEnv(gym.Env):
    
    def __init__(self):
        super(BD1GazeboEnv, self).__init__()
        
        self.name = "BD1 gazebo env"
        
        # MUST HAVE PARAMS
        self.env_interface_node_name = rospy.get_param("~env_interface_node_name", "")         
        self.req_actions = rospy.get_param("~actions", [])
        self.req_state = rospy.get_param("~state", [])   
        self.reward_type = rospy.get_param("~reward", "")  
        self.step_duration = rospy.get_param("~step_duration_sec", 0.1)
        
        # CONFIG
        rospy.wait_for_service(self.env_interface_node_name+'/configure')
        self.env_config_srv = rospy.ServiceProxy(self.env_interface_node_name+'/configure', Configure)
        
        self.config = self.env_config_srv(self.req_state, self.req_actions, self.reward_type)        
        if not self.config.configured:
            rospy.logerr("[{}] ERROR! Configure environment interface failed! Exit.".format(self.name))
            exit()
        
        self.action_space = spaces.Box(np.array([-1] * self.config.actions_dim), np.array([1] * self.config.actions_dim), dtype=np.float32)
        
        #high = np.array([np.inf] * self.config.state_dim)# state len!
        #self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array(self.config.state_high)
        low = np.array(self.config.state_low)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # services init
        rospy.wait_for_service(self.env_interface_node_name+'/reset')
        self.env_reset_srv = rospy.ServiceProxy(self.env_interface_node_name+'/reset', Reset)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
        
        rospy.wait_for_service(self.env_interface_node_name+'/step')
        self.env_step_srv = rospy.ServiceProxy(self.env_interface_node_name+'/step', Step)
        rospy.loginfo("[{}] step service ready!".format(self.name))

    def reset(self):
        state = np.array(self.env_reset_srv().state)
        return np.array(state).astype(np.float32)
    
    def step(self, action):
        srd = self.env_step_srv(self.step_duration, action.tolist())
        return np.array(srd.state), srd.reward, srd.done, {}
    
    def render(self, mode='console'):
        pass
    
    def close(self):
        pass
