import tensorflow as tf
import numpy as np

class RegressModel():
    def __init__(self,
                 env,
                 n_action,
                 n_state,
                 learning_rate=0.001,
                 ):
        self.env = env
        self.n_action = n_action
        self.n_state = n_state
        self.state = tf.placeholder(tf.float32, (None, n_state))
        self.actions = self._build_model(self.state)
        self.target_positions = tf.placeholder(tf.float32, (None, 2))
        self.pos_loss = self._position_loss(self.actions, self.target_positions)
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.pos_loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        action = self.sess.run(self.actions, feed_dict={self.state: state[np.newaxis, :]})
        return action[0]

    def learn(self, state, target):
        _, loss = self.sess.run([self.train, self.pos_loss], feed_dict={self.state: state, self.target_positions: target})
        return loss

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def _build_model(self, state):
        d1 = tf.keras.layers.Dense(64, activation='relu')(state)
        d2 = tf.keras.layers.Dense(64, activation='relu')(d1)
        actions = tf.keras.layers.Dense(self.n_action, activation='sigmoid')(d2)
        return actions

    def _position_loss(self, action, targets):
        theta0 = self.env.angle_range0[0] + (self.env.angle_range0[1] - self.env.angle_range0[0]) * action[:, 0]
        theta1 = self.env.angle_range1[0] + (self.env.angle_range1[1] - self.env.angle_range1[0]) * action[:, 1]
        theta2 = theta1

        x1 = self.env.point0.x + self.env.l0 * tf.cos(theta0)
        y1 = self.env.point0.y - self.env.l0 * tf.sin(theta0)

        x2 = x1 + self.env.l1 * tf.cos(theta1 + theta0)
        y2 = y1 - self.env.l1 * tf.sin(theta1 + theta0)

        x3 = x2 + self.env.l2 * tf.cos(theta2 + theta1 + theta0)
        y3 = y2 - self.env.l2 * tf.sin(theta2 + theta1 + theta0)

        x3 /= self.env.image_size[0]
        y3 /= self.env.image_size[1]

        pos = tf.stack([x3, y3], axis=-1)
        position_loss = tf.losses.mean_squared_error(targets, pos)
        return position_loss
