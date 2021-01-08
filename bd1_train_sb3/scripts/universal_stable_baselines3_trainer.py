#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import rospy
import numpy as np
import stable_baselines3 as sb3
from bd1_gazebo_env_interface import bd1_gazebo_gym_wrapper
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

class UniversalStableBaselines3Trainer(object):
    def __init__(self):
        rospy.init_node('universal_stable_baselines3_trainer')
        
        self.name = rospy.get_name()
        
        # load gym-wrapped env
        self.env = bd1_gazebo_gym_wrapper.BD1GazeboEnv()
        
        
        
        # load params
        self.model_dir = rospy.get_param('~model_dir','/tmp')
        self.task_name = rospy.get_param('~task_name','test')
        self.load_path = rospy.get_param('~load_path',None)
        self.checkpoint_freq = rospy.get_param('~checkpoint_freq',1000)
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
        
        self.action_noise_type = rospy.get_param('~action_noise_type', 0)        
        
        
        if self.action_noise_type == 1:
            self.proceed_noise()         
            self.action_noise = NormalActionNoise(self.action_noise_mean, self.action_noise_sigma)                                                     
        elif self.action_noise_type == 2:            
            self.proceed_noise()                                
            self.action_noise = OrnsteinUhlenbeckActionNoise(self.action_noise_mean, self.action_noise_sigma)                                                     
        else:
            self.action_noise = None
        
        ## ppo
        self.gae_lambda = rospy.get_param('~gae_lambda', 0.95)
        self.clip_range = rospy.get_param('~clip_range', 0.2)
        self.n_steps = rospy.get_param('~n_steps', 2048)
        ## sac
        self.tau = rospy.get_param('~tau', 0.005)
        self.buffer_size = rospy.get_param('~buffer_size', 1000000)
        self.use_sde = rospy.get_param('~use_sde', False)
        self.learning_starts = rospy.get_param('~learning_starts', 100)
        self.train_freq = rospy.get_param('~train_freq', 1)
        self.gradient_steps = rospy.get_param('~gradient_steps',1)
        self.n_episodes_rollout = rospy.get_param('~n_episodes_rollout', -1)
        
        self.save_dir = "{}/{}/{}/".format(self.model_dir, self.task_name, self.algorithm)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # wrap env to monitor (need for tensorboard?)
        self.env = Monitor(self.env, self.save_dir)
        
        # learning callbacks
        checkpoint_callback = CheckpointCallback(save_freq=self.checkpoint_freq, save_path=self.save_dir, name_prefix="checkpoint")
        bestrewardsave_callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=self.save_dir, verbose=1)
        self.callbacks = CallbackList([checkpoint_callback, bestrewardsave_callback])
        
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
                                    buffer_size = self.buffer_size,
                                    use_sde = self.use_sde,
                                    learning_starts = self.learning_starts,
                                    train_freq = self.train_freq,
                                    gradient_steps = self.gradient_steps,
                                    n_episodes_rollout = self.n_episodes_rollout,
                                    action_noise = self.action_noise)
                
            with open(self.save_dir + 'params.yaml', 'w') as file:
                yaml.dump(self.export_params(),file)
            
        else:
            #self.model.load(self.load_path)
            if self.algorithm == 'ppo':
                self.model = sb3.PPO.load(self.load_path)
            elif self.algorithm == 'sac':
                self.model = sb3.SAC.load(self.load_path)
                
    def proceed_noise(self):
        self.action_noise_mean = rospy.get_param('~action_noise_mean', 0)
        if isinstance(self.action_noise_mean, (int, float)):
            self.action_noise_mean = np.array( [self.action_noise_mean] * self.env.action_space.shape[0])
        elif isinstance(self.action_noise_mean, list):
            if len(self.action_noise_mean) != self.env.action_space.shape[0]:
                rospy.logerr("[{}] error! Given action noise mean has wrong length. Exit.".format(self.name))
                exit()
            self.action_noise_mean = np.array(self.action_noise_mean)
        
        self.action_noise_sigma = rospy.get_param('~action_noise_sigma', 1)
        if isinstance(self.action_noise_sigma, (int, float)):
            self.action_noise_sigma = np.array( [self.action_noise_sigma] * self.env.action_space.shape[0])
        elif isinstance(self.action_noise_sigma, list):
            if len(self.action_noise_sigma) != self.env.action_space.shape[0]:
                rospy.logerr("[{}] error! Given action noise sigma has wrong length. Exit.".format(self.name))
                exit()
            self.action_noise_sigma = np.array(self.action_noise_sigma)        
        
        
    def save_model(self, name):
        self.model.save(self.save_dir + name)
        rospy.logwarn("saved model {} to {}".format(name, self.save_dir))
        
    def export_params(self):
        params = {}
        params['algorithm'] = self.algorithm
        
        params['reward'] = rospy.get_param('~reward','')
        params['env_interface_node_name'] = rospy.get_param('~env_interface_node_name','')
        params['state'] = rospy.get_param('~state','')
        params['actions'] = rospy.get_param('~actions','')
        
        # common hyper
        params['policy'] = self.policy
        params['policy_kwargs'] = self.policy_kwargs
        params['total_timesteps'] = self.total_timesteps
        params['learning_rate'] = self.learning_rate
        params['gamma'] = self.gamma
        params['verbose'] = self.verbose
        params['batch_size'] = self.batch_size
        
        params['action_noise_type'] = self.action_noise_type
        params['action_noise_mean'] = self.action_noise_mean
        params['action_noise_sigma'] = self.action_noise_sigma
        
        # ppo
        if self.algorithm == 'ppo':
            params['n_epochs'] = self.n_epochs        
            params['gae_lambda'] = self.gae_lambda
            params['clip_range'] = self.clip_range
            params['n_steps'] = self.n_steps
        # sac
        if self.algorithm == 'sac':
            params['tau'] = self.tau
            params['buffer_size'] = self.buffer_size    
            params['use_sde'] = self.use_sde
            params['learning_starts'] = self.learning_starts
            params['train_freq'] = self.train_freq
            params['gradient_steps'] = self.gradient_steps
            params['n_episodes_rollout'] = self.n_episodes_rollout
            
            
        return params
        
    def run(self):
        if self.load_path is None:
            self.model.learn(total_timesteps = self.total_timesteps, tb_log_name="run", callback = self.callbacks)
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

# from https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_reward')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True

if __name__ == '__main__' :
    usbl3t = UniversalStableBaselines3Trainer()
    usbl3t.run()
    
