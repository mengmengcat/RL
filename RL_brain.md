# RL
TO build a universal fundamental DQN network which can be used with every beginners 
and this project is still updating~
hope some one can join me!



import tensorflow as tf
import numpy as np
class DeepQNetwork:
    #initialize parameter
    def  __init__(self,gamme,learning_rate,n_features,n_actions,n_layers,memory_size,batch_size):
        self.gamme = gamme
        self.lr = learning_rate
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.memory_size = memory_size
        self.loss_record = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size = batch_size
    #compute the Q values
    def q_eval_net(self):
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.memory = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
        self.w_initializer = tf.random_normal_initializer(0, 0.3)
        self.b_initializer = tf.constant_initializer(0.1)
        #layer1
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features,self.n_layers], initializer=self.w_initializer, collections=self.memory)
            b1 = tf.get_variable('b1', [1,self.n_layers], initializer=self.b_initializer, collections=self.memory)
            l1 = tf.nn.relu(tf.matmul(w1, self.s) + b1)
        #layer2
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [self.n_layers,self.n_actions], initializer=self.w_initializer, collections=self.memory)
            b2 = tf.get_variable('b2', [1,self.n_actions], initializer=self.b_initializer, collections=self.memory)
            self.q_eval = tf.matmul(w2, l1) + b2
    #compute the  Q next state values
    def q_next_net(self):
        self.s_ = tf.placeholder(tf.float32, [None,self.n_features], name='s_')
        # layer1
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features,self.n_layers], initializer=self.w_initializer, collections=self.memory)
            b1 = tf.get_variable('b1', [1,self.n_layers], initializer=self.b_initializer, collections=self.memory)
            l1 = tf.nn.relu(tf.matmul(w1, self.s_) + b1)
        # layer2
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [self.n_layers,self.n_actions], initializer=self.w_initializer, collections=self.memory)
            b2 = tf.get_variable('b2', [1,self.n_actions], initializer=self.b_initializer, collections=self.memory)
            self.q_next = tf.matmul(w2, l1) + b2
    # transform action position to match the q
    def transform(self):
        q_eval_zero = np.zeros(self.n_features)
        q_next_zero = np.zeros(self.n_features)
        q_eval_max = np.max(self.q_eval)
        q_next_max = np.max(self.q_next)
        q_eval_zero[0] = q_eval_max
        q_next_zero[0] = q_next_max
        self.q_eval = q_eval_zero
        self.q_next = q_next_zero

    def store_memory(self,s,a,r,s_):
        self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        if self.memory_counter >= self.memory_size:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
        else :
            self.memory[self.memory_counter,:] = transition
            self.memory_counter += 1
            #better code from Morvan Zhou :
            #index = self.memory_counter % self.memory_size
            #self.memory[index, :] = transition
            #self.memory_counter += 1

    def chocie(self,observation):
        observation = observation [np.newaxis,:]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action

    def learn(self,reward):
        sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next,q_eval = self.sess.run([self.q_next,self.q_eval],
                                      feed_dict = {self.s_: batch_memory[:, -self.n_features:],
                                                   self.s: batch_memory[:, :self.n_features]})
        self.q_target = reward + self.gamme * np.max(q_next)
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.loss_record.append(self.loss)

