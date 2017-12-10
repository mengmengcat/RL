# RL
TO build a universal fundamental DQN network which can be used with everydody



import tensorflow as tf

import numpy as np

class DeepQNetwork:

    #初始化参数
    def  __init__(self,gamme,learning_rate):
        self.gamme = gamme
        self.lr = learning_rate

    #Q值计算网络
    def q_eval_net(self):
        self.s = tf.placeholder(tf.float32,[],name='s')
        self.memory = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
        self.w_initializer = tf.random_normal_initializer(0, 0.3)
        self.b_initializer = tf.constant_initializer(0.1)
        #layer1
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [], initializer=self.w_initializer, collections=self.memory)
            b1 = tf.get_variable('b1', [], initializer=self.b_initializer, collections=self.memory)
            l1 = tf.nn.relu(tf.matmul(w1, self.s) + b1)
        #layer2
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [], initializer=self.w_initializer, collections=self.memory)
            b2 = tf.get_variable('b2', [], initializer=self.b_initializer, collections=self.memory)
            self.q_eval = tf.matmul(w2, l1) + b2
    #Q next state compute
    def q_next_net(self):
        self.s_ = tf.placeholder(tf.float32, [], name='s_')
        # layer1
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [], initializer=self.w_initializer, collections=self.memory)
            b1 = tf.get_variable('b1', [], initializer=self.b_initializer, collections=self.memory)
            l1 = tf.nn.relu(tf.matmul(w1, self.s_) + b1)
        # layer2
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [], initializer=self.w_initializer, collections=self.memory)
            b2 = tf.get_variable('b2', [], initializer=self.b_initializer, collections=self.memory)
            self.q_next = tf.matmul(w2, l1) + b2
    #action 位置匹配转换
    def transform(self):
        q_eval_zero = np.zeros()
        q_next_zero = np.zeros()
        q_eval_max = np.max(self.q_eval)
        q_next_max = np.max(self.q_next)
        q_eval_zero[0] = q_eval_max
        q_next_zero[0] = q_next_max
        self.q_eval = q_eval_zero
        self.q_next = q_next_zero


    def learn(self,reward):
        self.q_target = reward + self.gamme * np.max(self.q_next)
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
