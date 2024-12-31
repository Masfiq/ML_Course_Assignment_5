# Agent and TicTacToe classes

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import copy

import neuralnetworksA4 as nn
import rl_framework as rl  # for abstract classes rl.Environment and rl.Agent

######################################################################

class TicTacToe(rl.Environment):

    def __init__(self):
        self.state = None
        self.observation_size = 9
        self.action_size = 1
        self.player = 'X'

        self.observation_means = [0] * 9
        self.observation_stds = [0.8] * 9
        self.action_means = [4]
        self.action_stds =  [2.5]
        self.Q_means = [0]
        self.Q_stds = [1]
        
    def initialize(self):
        self.state = np.array([0] * 9)
        self.player = 'X'
        
    def act(self, action): 
        self.state = self.state.copy()
        self.state[action] = 1 if self.player == 'X' else -1
        self.player = 'X' if self.player == 'O' else 'O'

    def observe(self):
        return self.state

    def reinforcement(self):
        if self._won('X'):
            return 1
        if self._won('O'):
            return -1
        return 0

    def valid_actions(self):
        return np.where(self.state == 0)[0]

    def _won(self, player):
        marker = 1 if player == 'X' else -1  # flipped player because already set to next player
        combos = np.array((0,1,2, 3,4,5, 6,7,8, 0,3,6, 1,4,7, 2,5,8, 0,4,8, 2,4,6))
        return np.any(np.all(marker == self.state[combos].reshape((-1, 3)), axis=1))

    def _draw(self):
        return len(self.valid_actions()) == 0

    def terminal_state(self):
        return self._won('X') or self._won('O') or self._draw()

    def __str__(self):
        markers = np.array(['O', ' ', 'X'])
        s = '''
    {}|{}|{}
    -----
    {}|{}|{}
    ------
    {}|{}|{}'''.format(*markers[1 + self.state])
        return s

    def __repr__(self):
        return self.__str__()
        
######################################################################

class QnetAgent(rl.Agent):
    
    def __init__(self, environment, n_hiddens_each_layer, max_or_min='max'):
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.env = environment
        self.R_sign = 1 if max_or_min == 'max' else -1
        self.initialize()
        
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
            action = np.random.choice(actions)

        else:
            # Greedy Move
            np.random.shuffle(actions)
            obs = self.env.observe()
            Qs = np.array([self.use(np.hstack((obs, a))) for a in actions])
            action = actions[np.argmax(Qs)]

        return action

    def use(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.Qnet.use(X)

    def clear_samples(self):
        self.X = []
        self.R = []
        self.Done = []

    def add_sample(self, obs, action, r, done):
        self.X.append(np.hstack((obs, action)))
        self.R.append(r)
        self.Done.append(done)

    def update_Qn(self):

        env = self.env
        env.initialize()

        self.X = np.vstack(self.X)
        self.R = np.array(self.R).reshape(-1, 1)
        self.Done = np.array(self.Done).reshape(-1, 1)

        Qn = np.zeros_like(self.R)

        last_steps = np.where(self.Done)[0]
        first = 0
        for last_step in last_steps:
            Qn[first:last_step - 1] = self.use(self.X[first + 1:last_step])
            first = last_step

        return Qn 

    def train(self, n_epochs, method, learning_rate):
        gamma = 0.9

        for epoch in range(n_epochs):

            Qn = self.update_Qn()
            # R is from Player X's perspective. Negate it for Player O.
            T = self.R_sign * self.R + gamma * Qn
            self.Qnet.train(self.X, T, self.X, T,
                            n_epochs=1, method=method, learning_rate=learning_rate, verbose=False)
            
    def __str__(self):
        return self.Qnet.__str__()
    
    def __repr__(self):
        return self.__str__()

    


######################################################################

if __name__ == '__main__':

    class Game:

        def __init__(self, environment, agents):

            self.env = environment
            self.agents = agents

        def train(self, parms):

            n_batches = parms['n_batches']
            n_trials_per_batch = parms['n_trials_per_batch']
            n_epochs = parms['n_epochs']
            method = parms['method']
            learning_rate = parms['learning_rate']
            epsilon = parms['initial_epsilon']

            env = self.env

            final_epsilon = 0.001
            epsilon_decay =  np.exp((np.log(final_epsilon) - np.log(epsilon)) / (n_batches)) # to produce this final value

            epsilon_trace = []
            outcomes = []

            for batch in range(n_batches):
                agents['X'].clear_samples()
                agents['O'].clear_samples()

                for trial in range(n_trials_per_batch):

                    env.initialize()
                    done = False

                    while not done:

                        agent = agents[env.player]
                        obs = self.env.observe()
                        action = agent.epsilon_greedy(epsilon)

                        env.act(action)
                        r = env.reinforcement()
                        done = env.terminal_state()

                        # print(env)
                        # print(r)

                        agent.add_sample(obs, action, r, done)

                    outcomes.append(r)

                # end n_trials_per_batch
                self.agents['X'].train(n_epochs, method, learning_rate)
                self.agents['O'].train(n_epochs, method, learning_rate)

                epsilon_trace.append(epsilon)
                epsilon *= epsilon_decay

                if len(outcomes) % ((n_batches * n_trials_per_batch) // 20) == 0:
                    print(f'{len(outcomes)} games, {np.mean(outcomes):.2f} outcome mean')

            plt.figure(1)
            plt.clf()
            plt.subplot(3, 1, 1)
            n_per = 10
            n_bins = len(outcomes) // n_per
            outcomes_binned = np.array(outcomes).reshape(-1, n_per)
            avgs = outcomes_binned.mean(1)
            xs = np.linspace(n_per, n_per * n_bins, len(avgs))
            plt.plot(xs, avgs)
            plt.axhline(y=0, color='orange', ls='--')
            plt.ylabel('R')

            plt.subplot(3, 1, 2)
            plt.plot(xs, np.sum(outcomes_binned == -1, axis=1), 'r-', label='O Wins')
            plt.plot(xs, np.sum(outcomes_binned == 0, axis=1), 'b-', label='Draws')
            plt.plot(xs, np.sum(outcomes_binned == 1, axis=1), 'g-', label='X Wins')
            plt.legend(loc='center')
            plt.ylabel(f'Number of Games\nin Bins of {n_per:d}')

            plt.subplot(3, 1, 3)
            plt.plot(epsilon_trace)
            plt.ylabel(r'$\epsilon$')

            return outcomes, epsilon_trace


        def play_game(self):
            ttt = self.env
            agents = self.agents
            ttt.initialize()
            while True:
                agent = agents[env.player]
                obs = ttt.observe()
                action = agent.epsilon_greedy(epsilon=0.1)
                ttt.act(action)
                print(ttt)
                print(ttt.reinforcement)
                if ttt.terminal_state():
                    break

        def play_game_show_Q(self):
            ttt = self.env
            agents = self.agents
            plt.figure(2)
            plt.clf()
            step = 0

            ttt.initialize()
            while True:
                agent = agents[env.player]
                obs = ttt.observe()
                action = agent.epsilon_greedy(0.8 if step == 0 else 0.0)
                ttt.act(action)
                step += 1

                plt.subplot(5, 2, step)
                actions = ttt.valid_actions()
                Qs = np.array([agent.use((stack_sa(obs, a))) for a in actions])
                board_image = np.array([np.nan] * 9)
                for Q, a in zip(Qs, actions):
                    board_image[a] = Q
                board_image = board_image.reshape(3, 3)
                maxmag = np.nanmax(np.abs(board_image))
                print(f'{maxmag=}')
                plt.imshow(board_image, cmap='coolwarm', vmin=-maxmag, vmax=maxmag)
                plt.colorbar()
                # print(actions.ravel())
                # print(board_image)
                # plt.title(player)
                i = -1
                for row in range(3):
                    for col in range(3):
                        i += 1
                        if sn[i] == 1:
                            plt.text(col, row, 'X', ha='center',
                                     fontweight='bold', fontsize='large', color='black')
                        elif sn[i] == -1:
                            plt.text(col, row, 'O', ha='center',
                                     fontweight='bold', fontsize='large', color='black')
                plt.axis('off')
                if ttt.terminal_state():
                    break

            plt.tight_layout()
 



    ttt = TicTacToe()
    agents = {'X': QnetAgent(ttt, [20, 20], 'max'), 'O': QnetAgent(ttt, [], 'min')}
    game = Game(ttt, agents)

    parms = {
        'n_batches': 1000,
        'n_trials_per_batch': 10,
        'n_epochs': 3,
        'method': 'scg',
        'learning_rate': 0.2,
        'initial_epsilon': 1
    }

    outcomes = game.train(parms)

    game.play_game_show_Q()
