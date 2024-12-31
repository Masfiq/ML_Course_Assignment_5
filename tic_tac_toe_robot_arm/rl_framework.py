import numpy as np

from abc import ABC, abstractmethod
    
class Environment(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
        # ...

    @abstractmethod
    def initialize(self):
        self.state = None
        ...
        
    @abstractmethod
    def valid_actions(self):
        pass
        # return list of valid actions based on self.state
        
    @abstractmethod
    def observe(self):
        self.statte = self.state.copy()
        # ...
        # return what agent can observe about current self.state
    
    @abstractmethod
    def act(self, action):
        self.state = self.state.copy()
        # ...
    
    @abstractmethod
    def reinforcement(self):
        pass
        # r = some function of self.state
        # return r # scalar reinforcement
   
    def terminal_state(self, state):
        return False  # True if state is terminal state
    
    @abstractmethod
    def __str__(self):
        pass
        # return string to print current self.state
        
    def __repr__(self):
        return self.__str__()


class Agent(ABC):
    
    def __init__(self, environment, n_hiddens_each_layer, max_or_min='max'):
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.env = environment
        self.R_sign = 1 if max_or_min == 'max' else -1
        self.initialize()
        
    @abstractmethod
    def initialize(self):
        pass
    
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
        # env.initialize()

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

    @abstractmethod
    def train(self, n_epochs, method, learning_rate):
        gamma = 0.9

        for epoch in range(n_epochs):
            Qn = self.update_Qn()
            T = self.R_sign * self.R + gamma * Qn
            # ...Use following if neural net                                
            self.Qnet.train(self.X, T, self.X, T,
                            n_epochs=1, method=method, learning_rate=learning_rate, verbose=False)

    @abstractmethod
    def use(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return # calculate Q values for each row of X, each consisting of state and action


