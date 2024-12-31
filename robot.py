import numpy as np
import matplotlib.pyplot as plt
import itertools  # for product (cross product)
import random

import copy
from math import pi

import neuralnetworksA4 as nn
import rl_framework as rl  # for abstract classes rl.Environment and rl.Agent

class Robot(rl.Environment):

    def __init__(self):
        link_lengths = [4, 3]
        self.n_links = len(link_lengths)
        self.angles = []
        self.link_lengths = np.array(link_lengths)
        self.state = np.zeros(self.n_links)  # joint angles
        self.points = [[10, 10] for _ in range(self.n_links + 1)]
        self.lim = sum(link_lengths)
        self.update_points()
        self.goal = None
        # single_joint_actions = [-0.5, -0.1, 0.0, 0.1, 0.5]
        single_joint_actions = [-0.1, -0.05, 0.0, 0.05, 0.1]
        self.valid_action_values =  np.array(list(itertools.product(single_joint_actions, repeat=self.n_links)))

        self.observation_size = self.n_links * 2
        self.action_size = self.n_links
        self.observation_means = [0] * 2 * self.n_links
        self.observation_stds = [0.7] * 2 * self.n_links
        self.action_means = [0.0] * self.n_links
        self.action_stds =  [0.08] * self.n_links
        self.Q_means = [-5]
        self.Q_stds = [1.6]
        
        self.goal = [5., 5.]
        
    def initialize(self):
        self.state = np.random.uniform(-pi, pi, size=(self.n_links))
        self.previous_dist_to_goal = None
        
    def set_goal(self, goal):
        self.goal = goal

    def act(self, action): 
        self.state = self.state.copy()
        self.state += action   # joint_angle_deltas
        np.clip(self.state, 0.0, 2 * pi)
        # if self.record_angles:
        #     self.angles.append(self.joint_angles * 180 / pi)
        self.update_points()

    def observe(self):
        return np.hstack((np.sin(self.state),
                          np.cos(self.state)))

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = (self.points[i - 1][0]
                                 + self.link_lengths[i - 1] * np.cos(np.sum(self.state[:i])))
            self.points[i][1] = (self.points[i - 1][1] +
                                 self.link_lengths[i - 1] * np.sin(np.sum(self.state[:i])))
        self.end_effector = np.array(self.points[self.n_links]).T

    def set_points(self, points):
        self.points = points

    def reinforcement(self):
        dist_to_goal = np.sqrt(np.sum((self.goal - self.end_effector)**2))
        return -dist_to_goal

    def valid_actions(self):
        return self.valid_action_values

    def draw(self, alpha=1):
        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 'r-', alpha=alpha)
            plt.plot(self.points[i][0], self.points[i][1], 'k.', alpha=alpha)
        plt.axis('off')
        plt.axis('square')
        plt.xlim([-1, 21])
        plt.ylim([-1, 21])
        # plt.pause(1e-2)
        
    def __str__(self):
        s = f'Robot with joint angles {self.state}'
        return s

    def __repr__(self):
        return self.__str__()
        
######################################################################

class QnetAgent(rl.Agent):
    
    def initialize(self):
        env = self.env
        ni = env.observation_size + env.action_size
        self.Qnet = nn.NeuralNetwork(ni, self.n_hiddens_each_layer, 1)
        self.Qnet.X_means = np.array(env.observation_means + env.action_means)
        self.Qnet.X_stds = np.array(env.observation_stds + env.action_stds)
        self.Qnet.T_means = np.array(env.Q_means)
        self.Qnet.T_stds = np.array(env.Q_stds)

    def epsilon_greedy(self, epsilon):

        actions = self.env.valid_actions()

        if np.random.uniform() < epsilon:
            # Random Move
            action = random.choice(actions.tolist())
        else:
            # Greedy Move
            np.random.shuffle(actions)
            obs = self.env.observe()
            Qs = np.array([self.use(np.hstack((obs, a))) for a in actions])
            action = actions[np.argmax(Qs)]  # Minimize sum of distances to goal
        return action

    def use(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.Qnet.use(X)

    def train(self, n_epochs, method, learning_rate, gamma):

        for epoch in range(n_epochs):

            Qn = self.update_Qn()
            # print(f'{Qn.mean()=}')
            T = self.R + gamma * Qn
            # Remove samples with done=True, because no next state
            only_use_these = np.where(self.Done == False)[0]
            self.Qnet.train(self.X[only_use_these], T[only_use_these], self.X[only_use_these], T[only_use_these],
                            n_epochs=1, method=method, learning_rate=learning_rate, verbose=False)
            
    def __str__(self):
        return self.Qnet.__str__()
    
    def __repr__(self):
        return self.__str__()

######################################################################
######################################################################
######################################################################

if __name__ == '__main__':

######################################################################

    class Experiment:

        def __init__(self, environment, agent):

            self.env = environment
            self.agent = agent

        def train(self, parms):

            n_batches = parms['n_batches']
            # n_trials_per_batch = parms['n_trials_per_batch']
            n_steps_per_batch = parms['n_steps_per_batch']
            n_epochs = parms['n_epochs']
            method = parms['method']
            learning_rate = parms['learning_rate']

            env = self.env

            final_epsilon = 0.01
            epsilon_decay =  np.exp(np.log(final_epsilon) / (n_batches)) # to produce this final value
            epsilon = 1.0

            epsilon_trace = []
            outcomes = []

            for batch in range(n_batches):
                agent.clear_samples()
                env.initialize()

                for step in range(n_steps_per_batch):

                    obs = self.env.observe()
                    action = agent.epsilon_greedy(epsilon)

                    env.act(action)
                    r = env.reinforcement()

                    # print(r, obs, action)

                    done = step == n_steps_per_batch - 1
                    agent.add_sample(obs, action, r, done)

                outcomes.append(r)

                self.agent.train(n_epochs, method, learning_rate)

                epsilon_trace.append(epsilon)
                epsilon *= epsilon_decay

                # if len(outcomes) % ((n_batches * n_steps_per_batch) // 20) == 0:
                #     print(f'{len(outcomes)} trials, {np.mean(outcomes):.4f} outcome mean')
                if len(outcomes) % (n_batches // 20) == 0:
                    print(f'{len(outcomes)} batches, {np.mean(outcomes):.4f} outcome mean')

            plt.figure(1)
            plt.clf()
            plt.subplot(2, 1, 1)
            n_per = 10
            n_bins = len(outcomes) // n_per
            outcomes_binned = np.array(outcomes).reshape(-1, n_per)
            avgs = outcomes_binned.mean(1)
            xs = np.linspace(n_per, n_per * n_bins, len(avgs))
            plt.plot(xs, avgs)
            plt.axhline(y=0, color='orange', ls='--')
            plt.ylabel('R')

            plt.subplot(2, 1, 2)
            plt.plot(epsilon_trace)
            plt.ylabel(r'$\epsilon$')
            plt.pause(0.1)

            return outcomes, epsilon_trace

        def animate(self, n_steps):
            plt.clf()
            robot = self.env
            robot.initialize()
            agent = self.agent
            points = np.zeros((n_steps, robot.n_links + 1, 2))
            actions = np.zeros((n_steps, robot.n_links))
            Q_values = np.zeros((n_steps))

            for i in range(n_steps):
                action = agent.epsilon_greedy(epsilon=0.0)
                Q = agent.use(np.hstack((robot.observe(), action)))
                self.env.act(action)
                points[i] = robot.points
                actions[i] = action
                Q_values[i] = Q[0, 0]

            Q_min, Q_max = np.min(Q_values), np.max(Q_values)
            for i in range(n_steps):
                plt.clf()
                plt.scatter(robot.goal[0], robot.goal[1], s=40, c='blue')
                action = actions[i]
                robot.set_points(points[i])
                robot.draw(alpha=(Q_values[i] - Q_min) / (Q_max - Q_min))
 



    robot = Robot()
    robot.set_goal([5., 5.])
    
    agent = QnetAgent(robot, [50, 20, 20])
    experiment = Experiment(robot, agent)

    parms = {
        'n_batches': 1000,
        'n_steps_per_batch': 500,
        'n_epochs': 3,
        'method': 'scg',
        'learning_rate': 0.01
    }

    outcomes = experiment.train(parms)

    plt.figure(2)
    plt.clf()
    experiment.animate(100)

