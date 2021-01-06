#!/usr/bin/env python
# coding: utf-8

import rospy
import stable_baselines3 as sb3
from bd1_gazebo_env_interface import bd1_gazebo_gym_wrapper

class UniversalStableBaselines3Trainer(object):
    def __init__(self):
        rospy.init_node('universal_stable_baselines3_trainer')
        
        # load gym-wrapped env
        self.env = bd1_gazebo_gym_wrapper.BD1GazeboEnv()
        
        # load params
        self.model_dir = rospy.get_param('~model_dir','/tmp')
        self.task_name = rospy.get_param('~task_name','test')
        self.load_path = rospy.get_param('~load_path',None)
        # load hypers
        ## common
        self.algorithm = rospy.get_param('~algorithm', '').lower()
        self.policy = rospy.get_param('~policy', 'MlpPolicy')
        self.policy_kwargs = rospy.get_param('~policy_kwargs', {})
        print(self.policy_kwargs)
        self.total_timesteps = rospy.get_param('~total_timesteps',10000)
        self.learning_rate = rospy.get_param('~learning_rate',0.0003)
        self.gamma = rospy.get_param('~gamma', 0.99)
        self.verbose = rospy.get_param('~verbose',0)
        self.batch_size = rospy.get_param('~batch_size', 64)
        self.n_epochs = rospy.get_param('~n_epochs', 10)
        ## ppo
        self.gae_lambda = rospy.get_param('~gae_lambda', 0.95)
        self.clip_range = rospy.get_param('~clip_range', 0.2)
        self.n_steps = rospy.get_param('~n_steps', 2048)
        ## sac
        self.tau = rospy.get_param('~tau', 0.005)
        self.buffer_size = rospy.get_param('~buffer_size', 1000000)
        
        
        self.save_dir = "{}/{}/{}/".format(self.model_dir, self.task_name, self.algorithm)
        
        if self.load_path is None:
            if self.algorithm == 'ppo':
                self.model = sb3.PPO(self.policy, 
                                    self.env, 
                                    verbose=self.verbose, 
                                    learning_rate = self.learning_rate, 
                                    gamma = self.gamma, 
                                    gae_lambda = self.gae_lambda, 
                                    clip_range = self.clip_range, 
                                    batch_size = self.batch_size, 
                                    tensorboard_log=self.save_dir,
                                    n_steps = self.n_steps,
                                    n_epochs = self.n_epochs)
            #elif self.algorithm == 'a3c':
                #self.model = sb3.A3C()
            elif self.algorithm == 'sac':
                self.model = sb3.SAC(self.policy, 
                                    self.env, 
                                    verbose=self.verbose, 
                                    tensorboard_log=self.save_dir, 
                                    policy_kwargs = self.policy_kwargs, 
                                    learning_rate=self.learning_rate, 
                                    gamma=self.gamma, 
                                    batch_size= self.batch_size, 
                                    tau = self.tau,
                                    buffer_size = self.buffer_size)
            
        else:
            #self.model.load(self.load_path)
            if self.algorithm == 'ppo':
                self.model = sb3.PPO.load(self.load_path)
            elif self.algorithm == 'sac':
                self.model = sb3.SAC.load(self.load_path)
        
    def save_model(self, name):
        self.model.save(self.save_dir + name)
        rospy.logwarn("saved model {} to {}".format(name, self.save_dir))
        
    def export_params(self):
        params = {}
        params['algorithm'] = self.algorithm
        params['policy'] = self.policy
        params['policy_kwargs'] = self.policy_kwargs
        params['total_timesteps'] = self.total_timesteps
        params['learning_rate'] = self.learning_rate
        params['gamma'] = self.gamma
        params['verbose'] = self.verbose
        params['batch_size'] = self.batch_size
        params['n_epochs'] = self.n_epochs
        # ppo
        params['gae_lambda'] = self.gae_lambda
        params['clip_range'] = self.clip_range
        params['n_steps'] = self.n_steps
        
        return params
        
    def run(self):
        if self.load_path is None:
            self.model.learn(total_timesteps = self.total_timesteps, tb_log_name="run")
            self.save_model('fully_trained')
        else:
            obs = self.env.reset()
            rospy.logwarn("testing...")
            while not rospy.is_shutdown():
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                if done:
                    obs = self.env.reset()
                    rospy.logwarn("new episode started")
                    
        #rospy.spin()
        #exit()


if __name__ == '__main__' :
    usbl3t = UniversalStableBaselines3Trainer()
    usbl3t.run()
    
