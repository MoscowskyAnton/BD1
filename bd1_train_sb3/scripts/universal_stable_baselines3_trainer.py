#!/usr/bin/env python
# coding: utf-8

import rospy
import stable_baselines3 as sb3
from bd1_gazebo_env_interface import bd1_gazebo_gym_wrapper#BD1GazeboEnv

class UniversalStableBaselines3Trainer(object):
    def __init__(self):
        rospy.init_node('universal_stable_baselines3_trainer')
        
        # load gym-wrapped env
        self.env = bd1_gazebo_gym_wrapper.BD1GazeboEnv()
        
        # load hypers
        self.algorithm = rospy.get_param('~algorithm', '').lower()
        self.policy = rospy.get_param('~policy', 'MlpPolicy')
        self.total_timesteps = rospy.get_param('~total_timesteps',10000)
        
        if self.algorithm == 'ppo':
            self.model = sb3.PPO(self.policy, self.env, verbose=1)
        elif self.algorithm == 'a3c':
            self.model = sb3.A3C()
        
        
    def run(self):
        self.model.learn(total_timesteps = self.total_timesteps)
        
        rospy.spin()
        


if __name__ == '__main__' :
    usbl3t = UniversalStableBaselines3Trainer()
    usbl3t.run()
    
