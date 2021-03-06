import gym
import numpy as np
import math
from sklearn.preprocessing import KBinsDiscretizer


"""Q-learning algorithm with inspiration from https://www.youtube.com/watch?v=JNKvJEzuNsc&ab_channel=RichardBrooker 
and https://github.com/init-22/CartPole-v0-using-Q-learning-SARSA-and-DNN/blob/master/Qlearning_for_cartpole.py """

class CartPoleQAgent():
    def __init__(self, buckets=(6, 12), num_episodes = 500, min_lr = 0.1, min_epsilon = 0.1, discount = 1.0, decay = 25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        
        self.env = gym.make('CartPole-v0')

    
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

        self.upper_bounds = [ self.env.observation_space.high[2], math.radians(50)]
        self.lower_bounds = [ self.env.observation_space.low[2], -math.radians(50)]

    def discretize_state(self, obs): 
        """From continous state to discrete"""
        _, _, angle, angle_velocity = obs
        est = KBinsDiscretizer(n_bins=self.buckets,
                               encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([[angle, angle_velocity]])[0]))

    def choose_action(self, state):
        if (np.random.random() < self.epsilon): 
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_table[state])

    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += self.learning_rate * (reward + self.discount * np.max(self.Q_table[new_state]) - self.Q_table[state][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state

        print('Finished training!')

    def run(self):
        self.env = gym.wrappers.Monitor(self.env,'cartpole', force=True)
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
            
        return t   
            


if __name__ == "__main__":
    agent = CartPoleQAgent()
    agent.train()
    t = agent.run()
    print("Time", t)
