# Based on
# https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_PPO.py
#
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
import os

class PPO(object):
    """
    PPO class
    """
    def __init__(self, state_dim, action_dim, action_bound, hyperparams, method='clip'):
        self.hyperparams = hyperparams
        # critic
        with tf.name_scope('critic'):
            inputs = tl.layers.Input([None, state_dim], tf.float32, 'state')
            layer = tl.layers.Dense(self.hyperparams["CRITIC_LAYER1_SIZE"], tf.nn.relu)(inputs)
            layer = tl.layers.Dense(self.hyperparams["CRITIC_LAYER2_SIZE"], tf.nn.relu)(layer)
            layer = tl.layers.Dense(self.hyperparams["CRITIC_LAYER2_SIZE"], tf.nn.relu)(layer)
            layer = tl.layers.Dense(self.hyperparams["CRITIC_LAYER3_SIZE"], tf.nn.relu)(layer)
            v = tl.layers.Dense(1)(layer)
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # actor
        with tf.name_scope('actor'):
            inputs = tl.layers.Input([None, state_dim], tf.float32, 'state')
            layer = tl.layers.Dense(self.hyperparams["ACTOR_LAYER1_SIZE"], tf.nn.relu)(inputs)
            layer = tl.layers.Dense(self.hyperparams["ACTOR_LAYER2_SIZE"], tf.nn.relu)(layer)
            layer = tl.layers.Dense(self.hyperparams["ACTOR_LAYER2_SIZE"], tf.nn.relu)(layer)
            layer = tl.layers.Dense(self.hyperparams["ACTOR_LAYER3_SIZE"], tf.nn.relu)(layer)
            a = tl.layers.Dense(action_dim, tf.nn.tanh)(layer)
            mean = tl.layers.Lambda(lambda x: x * action_bound, name='lambda')(a)
            logstd = tf.Variable(np.ones(action_dim, dtype=np.float32) * self.hyperparams["LOGSTD"])
        self.actor = tl.models.Model(inputs, mean)
        self.actor.trainable_weights.append(logstd)
        self.actor.logstd = logstd
        print("Init logstd", logstd)
        self.actor.train()

        self.actor_opt = tf.optimizers.Adam(self.hyperparams["LR_A"])
        self.critic_opt = tf.optimizers.Adam(self.hyperparams["LR_C"])

        self.method = method
        if method == 'penalty':
            self.kl_target = self.hyperparams["KL_TARGET"]
            self.lam = self.hyperparams["LAM"]
        elif method == 'clip':
            self.epsilon = self.hyperparams["EPSILON"]

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound

    #def train_actor(self, state, action, adv, old_pi):
    def train_actor(self):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        s_all = np.array(self.state_buffer, np.float32)
        a_all = np.array(self.action_buffer, np.float32)
        r_all = np.array(self.cumulative_reward_buffer, np.float32)
        
        batch_size = s_all.shape[0]
        
        for _ in range(self.hyperparams["ACTOR_UPDATE_STEPS"]):
            minibatch_ind = np.random.randint(0, batch_size, self.hyperparams["MINIBATCH_SIZE"])
            
            state = s_all[minibatch_ind,:]
            action = a_all[minibatch_ind,:]
            reward = r_all[minibatch_ind,:]
            
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            old_pi = tfp.distributions.Normal(mean, std)
            adv = reward - self.critic(state)
            
            with tf.GradientTape() as tape:
                mean, std = self.actor(state), tf.exp(self.actor.logstd)
                pi = tfp.distributions.Normal(mean, std)

                ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
                surr = ratio * adv
                if self.method == 'penalty':  # ppo penalty
                    kl = tfp.distributions.kl_divergence(old_pi, pi)
                    kl_mean = tf.reduce_mean(kl)
                    loss = -(tf.reduce_mean(surr - self.lam * kl))
                else:  # ppo clip
                    loss = -tf.reduce_mean(
                        tf.minimum(surr,
                                tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                    )
            a_gard = tape.gradient(loss, self.actor.trainable_weights)
            self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method == 'penalty':
            return kl_mean

    #def train_critic(self, reward, state):
    def train_critic(self):
        """
        Update actor network
        :param reward: cumulative reward batch
        :param state: state batch
        :return: None
        """
        #reward = np.array(reward, dtype=np.float32)
        s_all = np.array(self.state_buffer, np.float32)        
        r_all = np.array(self.cumulative_reward_buffer, np.float32)
        
        batch_size = s_all.shape[0]
        
        for _ in range(self.hyperparams["CRITIC_UPDATE_STEPS"]):
            minibatch_ind = np.random.randint(0, batch_size, self.hyperparams["MINIBATCH_SIZE"])
            
            state = s_all[minibatch_ind,:]            
            reward = r_all[minibatch_ind,:]
        
            with tf.GradientTape() as tape:
                advantage = reward - self.critic(state)
                loss = tf.reduce_mean(tf.square(advantage))
            grad = tape.gradient(loss, self.critic.trainable_weights)
            self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        #s_all = np.array(self.state_buffer, np.float32)
        #a_all = np.array(self.action_buffer, np.float32)
        #r_all = np.array(self.cumulative_reward_buffer, np.float32)
        
        #batch_size = r_all.size[0]
        #minibatch_ind = np.random.randint(0, batch_size, self.hyperparams["MINIBATCH_SIZE"])
        
        #s = s_all[minibatch_ind,:]
        #a = a_all[minibatch_ind,:]
        #r = r_all[minibatch_ind,:]
        
        #mean, std = self.actor(s), tf.exp(self.actor.logstd)
        #pi = tfp.distributions.Normal(mean, std)
        #adv = r - self.critic(s)

        # update actor
        if self.method == 'penalty':
            for _ in range(self.hyperparams["ACTOR_UPDATE_STEPS"]):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            #for _ in range(self.hyperparams["ACTOR_UPDATE_STEPS"]):
                #self.train_actor(s, a, adv, pi)
            self.train_actor()

        # update critic
        #for _ in range(self.hyperparams["CRITIC_UPDATE_STEPS"]):
            #self.train_critic(r, s)
        self.train_critic()

        #self.state_buffer.clear()
        #self.action_buffer.clear()
        #self.cumulative_reward_buffer.clear()
        #self.reward_buffer.clear()

    def get_action(self, state, greedy=False):
        """
        Choose action
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        state = state[np.newaxis, :].astype(np.float32)
        mean, std = self.actor(state), tf.exp(self.actor.logstd)
        #print("Action logstd", self.actor.logstd)
        if greedy:
            action = mean[0]
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        return np.clip(action, -self.action_bound, self.action_bound)

    def save(self, path):
        """
        save trained weights
        :return: None
        """
        #path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)

    def load(self, path):
        """
        load trained weights
        :return: None
        """
        #path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        
        
    # aka GAE
    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + self.hyperparams["GAMMA"] * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()
