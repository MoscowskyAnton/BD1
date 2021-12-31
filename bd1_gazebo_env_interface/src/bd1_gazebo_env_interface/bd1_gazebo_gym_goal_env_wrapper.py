#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
import numpy as np
import gym
from gym import spaces
from bd1_gazebo_env_interface.srv import GoalEnvStep, GoalEnvReset, GoalEnvConfigure


'''
gym GoalEnv - https://github.com/openai/gym/blob/3394e245727c1ae6851b504a50ba77c73cd4c65b/gym/core.py#L160
highway parking - https://github.com/eleurent/highway-env/blob/master/highway_env/envs/parking_env.py
'''
class BD1GazeboGoalEnv(gym.GoalEnv):
    
    def __init__(self, max_episode_timesteps = 0):
        super(BD1GazeboGoalEnv, self).__init__()
        
        self.name = "BD1 gazebo goal env"
        
        self.max_episode_timesteps = max_episode_timesteps
        self.timesteps_counter = 0
        
        # MUST HAVE PARAMS
        self.env_interface_node_name = rospy.get_param("~env_interface_node_name", "")         
        self.req_actions = rospy.get_param("~actions", [])
        self.req_state = rospy.get_param("~state", [])   
        #self.reward_type = rospy.get_param("~reward", "")  
        self.req_goal = rospy.get_param("~goal", [])
        self.step_duration = rospy.get_param("~step_duration_sec", 0.1)
        
        self.reward_coeffs = np.array(rospy.get_param("~reward_coeffs", [])).astype(np.float32)
        self.p = rospy.get_param("~reward_p", 0.5)
        
        
        # CONFIG
        rospy.wait_for_service(self.env_interface_node_name+'/configure')
        self.env_config_srv = rospy.ServiceProxy(self.env_interface_node_name+'/configure', GoalEnvConfigure)
        
        self.config = self.env_config_srv(self.req_state, self.req_actions, self.req_goal)        
        if not self.config.configured:
            rospy.logerr("[{}] ERROR! Configure environment interface failed! Exit.".format(self.name))
            exit()
            
        if self.reward_coeffs.shape[0] != self.config.goal_dim:
            rospy.logerr(f"[{self.name}] different len of goal_dim and goal_coeffs!")
            exit()
        
        self.action_space = spaces.Box(np.array([-1] * self.config.actions_dim), np.array([1] * self.config.actions_dim), dtype=np.float32)
        
        #high = np.array([np.inf] * self.config.state_dim)# state len!
        #self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        s_high = np.array(self.config.state_high)
        s_low = np.array(self.config.state_low)
        g_high = np.array(self.config.goal_high)
        g_low = np.array(self.config.goal_low)
        #self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(g_low, g_high, dtype=np.float32),
                achieved_goal=spaces.Box(g_low, g_high, dtype=np.float32),
                observation=spaces.Box(s_low, s_high, dtype=np.float32),
            ))
        
        
        # services init
        rospy.wait_for_service(self.env_interface_node_name+'/reset')
        self.env_reset_srv = rospy.ServiceProxy(self.env_interface_node_name+'/reset', GoalEnvReset)
        rospy.loginfo("[{}] reset service ready!".format(self.name))
        
        rospy.wait_for_service(self.env_interface_node_name+'/step')
        self.env_step_srv = rospy.ServiceProxy(self.env_interface_node_name+'/step', GoalEnvStep)
        rospy.loginfo("[{}] step service ready!".format(self.name))
        
    def srv_to_gym_state(self, srv_msg):
        state = {}#gym.spaces.Dict()
        state['observation'] = np.array(srv_msg.observation).astype(np.float32)
        state['achieved_goal'] = np.array(srv_msg.achieved_goal).astype(np.float32)
        state['desired_goal'] = np.array(srv_msg.desired_goal).astype(np.float32)
        return state

    def reset(self):
        if self.max_episode_timesteps > 0:
            self.timesteps_counter = 0
        srv_res = self.env_reset_srv()
        
        return self.srv_to_gym_state(srv_res)
        
    
    def step(self, action):
        srd = self.env_step_srv(self.step_duration, action.tolist())
        state = self.srv_to_gym_state(srd)
        if( self.max_episode_timesteps > 0 ):
            self.timesteps_counter += 1
            if self.timesteps_counter >= self.max_episode_timesteps:
                return state, self._reward(state), True, {}
            else:
                return state, self._reward(state), srd.done, {}
        else:
            return state, self._reward(state), srd.done, {}
    
    def _reward(self, state):
        return self.compute_reward(state['achieved_goal'], state['desired_goal'], None)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        #print(type(achieved_goal), type(desired_goal))
        #reward = 1-np.power(np.dot(np.abs(achieved_goal - desired_goal) , self.reward_coeffs), self.p)
        reward = np.sum(desired_goal - np.abs(achieved_goal - desired_goal))
        #print(achieved_goal, desired_goal, achieved_goal - desired_goal, reward)
        return reward
    
    def render(self, mode='console'):
        pass
    
    def close(self):
        pass
