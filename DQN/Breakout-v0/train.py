# author: Shuming Fang
# email: fangshuming519@gmail.com
import numpy as np
from PIL import Image
import tensorflow as tf
import gym
env = gym.make('Breakout-v0')
n_actions = 4
from hyperparams import Hyperparams as hp

def resize(image):
    image = image[50:,:,:]
    image = Image.fromarray(image)
    image = image.convert('L')
    image = image.resize([84,84])
    image = np.array(image)
    image = image/255.0-0.5
    return image
    
table_s = np.random.randn(hp.capacity,hp.height,hp.width,hp.skip_channels)
table_s_ = np.random.randn(hp.capacity,hp.height,hp.width,hp.skip_channels)
table_r = np.random.randn(hp.capacity,)
table_a = np.array([np.random.randint(0,4) for i in range(hp.capacity)])
table_s = table_s.astype(np.float32)
table_s_ = table_s_.astype(np.float32)
table_r = table_r.astype(np.float32)
table_a = table_a.astype(np.int32)

class Graph:
    def __init__(self, training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if training:
                self.table_s = tf.placeholder(tf.float32,shape=(None,hp.height,hp.width,hp.skip_channels))
                self.table_s_ = tf.placeholder(tf.float32,shape=(None,hp.height,hp.width,hp.skip_channels))
                self.table_r = tf.placeholder(tf.float32,shape=(None,))
                self.table_a = tf.placeholder(tf.int32,shape=(None,))
            else:
                self.table_s = tf.placeholder(tf.float32,shape=(None,hp.height,hp.width,hp.skip_channels))

            with tf.variable_scope('eval_net'):
                self.q_eval = tf.layers.conv2d(self.table_s, filters=16,kernel_size=8,strides=(4,4),padding='valid',activation=tf.nn.relu)
                self.q_eval = tf.layers.conv2d(self.q_eval, filters=32,kernel_size=4,strides=(2,2),padding='valid',activation=tf.nn.relu)
                self.q_eval = tf.layers.flatten(self.q_eval)
                self.q_eval = tf.layers.dense(self.q_eval,256,activation=tf.nn.relu)
                self.q_eval = tf.layers.dense(self.q_eval,4)
            with tf.variable_scope('target_net'):
                self.q_next = tf.layers.conv2d(self.table_s_, filters=16,kernel_size=8,strides=(4,4),padding='valid',activation=tf.nn.relu)
                self.q_next = tf.layers.conv2d(self.q_next, filters=32,kernel_size=4,strides=(2,2),padding='valid',activation=tf.nn.relu)
                self.q_next = tf.layers.flatten(self.q_next)
                self.q_next = tf.layers.dense(self.q_next,256,activation=tf.nn.relu)
                self.q_next = tf.layers.dense(self.q_next,4)
            with tf.variable_scope('q_target'):
                q_target = self.table_r + hp.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.table_a)[0], dtype=tf.int32), self.table_a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(hp.lr).minimize(self.loss)
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            
       
if __name__ == '__main__':
    g = Graph();
    with g.graph.as_default():
        sv = tf.train.Supervisor(save_model_secs=0)
        with sv.managed_session() as sess:
            #step 1, offline learning
            counter = 0
            _ = sess.run(g.target_replace_op,feed_dict={g.table_s:table_s[:1],g.table_s_:table_s_[:1],g.table_r:table_r[:1],g.table_a:table_a[:1]})
            for i in range(1000):
                observation = env.reset()
                observation = resize(observation)
                observation_video = np.random.randn(hp.height, hp.width, hp.skip_channels)
                observation_video_ = np.random.randn(hp.height, hp.width, hp.skip_channels)
                for k in range(hp.skip_channels):
                    observation_video[:,:,k] = observation
                done = False
                lives = 5
                while not done:
                    env.render()
                    q_predict = sess.run(g.q_eval,feed_dict={g.table_s:observation_video[np.newaxis,:]})
                    action = np.argmax(q_predict)
                    if np.random.uniform()<hp.epsilon:
                        observation_, reward, done, info = env.step(action)
                        observation_ = resize(observation_)
                    else:
                        action = np.random.randint(0, n_actions)
                        observation_, reward, done, info = env.step(action)
                        observation_ = resize(observation_)
                    if info['ale.lives']!=lives:
                        reward = -1.0
                        lives = lives-1
                    if reward==1.0:
                        reward = 10.0
                    observation_video_[:,:,0:3] = observation_video_[:,:,1:4]
                    observation_video_[:,:,3] = observation_
                    table_s[:-1,:,:,:] = table_s[1:,:,:,:]
                    table_s[-1,:,:,:] = observation_video
                    table_s_[:-1,:,:,:] = table_s_[1:,:,:,:]
                    table_s_[-1,:,:,:] = observation_video_                       
                    table_r[:-1] = table_r[1:]
                    table_r[-1] = reward
                    table_a[:-1] = table_a[1:]
                    table_a[-1] = action
                    observation_video[:,:,0:3] = observation_video[:,:,1:4]
                    observation_video[:,:,3] = observation_
                    counter+=1
                    if counter%hp.capacity==0:
                        idx = np.random.randint(0, hp.capacity, hp.batch_size)
                        table_sample_s = table_s[idx]
                        table_sample_s_ = table_s_[idx]
                        table_sample_r = table_r[idx]
                        table_sample_a = table_a[idx]
                        if np.sum(table_r[table_r==10.0])>0:
                            exp_idx = np.where(table_r==1.0)[0]
                            table_sample_s[0:len(exp_idx)] = table_s[exp_idx]
                            table_sample_s_[0:len(exp_idx)] = table_s_[exp_idx]
                            table_sample_r[0:len(exp_idx)] = table_r[exp_idx]
                            table_sample_a[0:len(exp_idx)] = table_a[exp_idx]
                        for j in range(4):
                            _,loss = sess.run([g._train_op,g.loss],feed_dict={g.table_s:table_sample_s,g.table_s_:table_sample_s_,g.table_r:table_sample_r,g.table_a:table_sample_a})
                            print('Epsilon: ',i, 'Step: ',j, 'Loss: ', loss)
                        _ = sess.run(g.target_replace_op,feed_dict={g.table_s:table_sample_s,g.table_s_:table_sample_s_,g.table_r:table_sample_r,g.table_a:table_sample_a})


