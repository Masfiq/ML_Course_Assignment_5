import numpy as np
import matplotlib.pyplot as plt
import itertools  # for product (cross product)
import random
import time

# import Box2D.b2 as b2
import Box2D.b2 as b2

import copy
from math import pi

import cartpole_play_v1 as cp

import neuralnetworksA4 as nn

import rl_framework as rl  # for abstract classes rl.Environment and rl.Agent

import Box2D.b2 as b2
from math import pi
import cartpole_play as cp
import rl_framework as rl

class CartPole(rl.Environment):
    def __init__(self):
        # Initialize cart-pole system using the CartPole class from cartpole_play
        self.cartpole = cp.CartPole()
        self.valid_action_values = [-1, 0, 1]  # Actions: push left, no push, push right
        self.observation_size = 4  # Observations: x (position), xdot (velocity), a (angle), adot (angular velocity)
        self.action_size = 1  # Single action dimension
        self.observation_means = [0, 0, 0, 0]  # Initial means for observations
        self.observation_stds = [1, 1, 1, 1]   # Placeholder values for standard deviations
        self.action_means = [0.0]  # Mean for action normalization
        self.action_stds = [0.1]   # Standard deviation for action normalization
        self.Q_means = [0.5 * pi]  # Target mean for Q-values
        self.Q_stds = [2]          # Target standard deviation for Q-values
        self.force = 0.15  # Define the force magnitude for the cart
        
        # Initialize Box2D world with gravity
        self.world = b2World(gravity=(0, -9.8), doSleep=True)

        # Define a cart body in the Box2D world and directly attach a polygon fixture to it
        self.cart = self.world.CreateDynamicBody(
            position=(0, 2)  # Starting position of the cart
        )
        
        # Define and add shape directly to the cart body
        cart_shape = b2.b2PolygonShape(box=(0.5, 0.1))  # Dimensions (width, height) of the cart
        self.cart.CreatePolygonFixture(shape=cart_shape, density=1.0, friction=0.3)

        # Other attributes as before
        self.force = 0.15
        self.timeStep = 1.0 / 60.0
        self.velocityIterations = 6
        self.positionIterations = 2


    def initialize(self):
        # Reset cart-pole state for a new episode with randomized cart position and zero velocities/angles
        self.cartpole.cart.position[0] = np.random.uniform(-2., 2.)
        self.cartpole.cart.linearVelocity[0] = 0.0
        self.cartpole.pole.angle = 0  # Start with the pole hanging vertically
        self.cartpole.pole.angularVelocity = 0.0

    def reinforcement(self):
        # Reward function based on the pole's angle to encourage balance
        state = self.observe()
        angle_magnitude = np.abs(state[2])

        # Reward scheme based on angle, with zero being balanced
        if angle_magnitude > pi * 0.75:
            return -1  # Penalty for large deviation
        elif angle_magnitude < pi * 0.25:
            return 1  # Reward for small deviation (close to balanced)
        else:
            return 0  # Neutral reward
        
    def apply_force(self, action):
        # Apply a force to the cart based on the action
        # Here, `action` is expected to be -1 (left), 0 (no push), or 1 (right)
        if action == -1:
            force = -self.force  # Push left
        elif action == 1:
            force = self.force   # Push right
        else:
            force = 0            # No push
        
        # Assume self.cart is the cart object in the simulation
        self.cart.ApplyForceToCenter((force, 0), wake=True)
        
    def step(self):
        # Advance the Box2D world by one time step
        self.world.Step(self.timeStep, self.velocityIterations, self.positionIterations)
        self.world.ClearForces()  # Clear forces to avoid accumulation

    def act(self, action):
        # Apply action to the cart-pole system and step the simulation forward
        self.apply_force(action)
        self.cartpole.step()  # Advance the cart-pole simulation by one step

    def observe(self):
        # Return the current state as an observation array
        return np.array([
            self.cartpole.cart.position[0],
            self.cartpole.cart.linearVelocity[0],
            self.cartpole.pole.angle,
            self.cartpole.pole.angularVelocity
        ])

    def valid_actions(self):
        # Return the list of valid actions
        return self.valid_action_values

    def terminal_state(self, state):
        # Define terminal state (for simplicity, no terminal state in this setup)
        return False

    def __str__(self):
        # Return a string representation of the cart-pole's current state
        state = self.observe()
        return f"Cart Position: {state[0]}, Cart Velocity: {state[1]}, Pole Angle: {state[2]}, Pole Angular Velocity: {state[3]}"
    
    

class QnetAgent(rl.Agent):
    def __init__(self, environment, n_hiddens_each_layer, epsilon=0.1, learning_rate=0.01, gamma=0.99):
        # Initialize the QnetAgent with its environment and network configuration
        self.env = environment
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.epsilon = epsilon  # Epsilon for exploration in epsilon-greedy policy
        self.learning_rate = learning_rate  # Learning rate for updates
        self.gamma = gamma  # Discount factor

    def initialize(self):
        # Set up Q-network with the required input and output sizes
        env = self.env
        ni = env.observation_size + env.action_size  # Input size for the network
        self.Qnet = nn.NeuralNetwork(ni, self.n_hiddens_each_layer, 1)  # Q-network with single output for Q-value
        self.Qnet.X_means = np.array(env.observation_means + env.action_means)  # Means for normalization
        self.Qnet.X_stds = np.array(env.observation_stds + env.action_stds)  # Standard deviations for normalization
        self.Qnet.T_means = np.array(env.Q_means)  # Target means for Q-values
        self.Qnet.T_stds = np.array(env.Q_stds)    # Target standard deviations for Q-values

    def epsilon_greedy(self, state, epsilon=None):
        # Select an action using epsilon-greedy policy
        if epsilon is None:
            epsilon = self.epsilon  # Use default epsilon if none provided
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.valid_action_values)  # Explore: choose random action
        else:
            q_values = [self.q_value(state, action) for action in self.env.valid_action_values]
            return self.env.valid_action_values[np.argmax(q_values)]  # Exploit: choose best action

    def q_value(self, state, action):
        # Compute Q-value for a given state-action pair
        sa = np.hstack((state, [action]))  # Concatenate state and action as input
        return self.Qnet.use(sa)[0]  # Get Q-value from neural network

    def update_q_value(self, state, action, reward, next_state):
        # Update the Q-value for the state-action pair based on observed reward and next state
        current_q = self.q_value(state, action)
        next_q_values = [self.q_value(next_state, a) for a in self.env.valid_action_values]
        max_next_q = max(next_q_values)

        # Target for the Q-network
        target = reward + self.gamma * max_next_q
        target = np.array([target])  # Convert to array to match network input format

        # Input to the Q-network
        sa = np.hstack((state, [action]))  # State-action pair as input

        # Update Q-network weights using gradient descent
        self.Qnet.train(sa, target, n_epochs=1, method='sgd', learning_rate=self.learning_rate)

    def train(self, n_epochs, method, learning_rate, gamma):
        # Additional training logic if needed, for example, over multiple samples
        pass  # Placeholder for additional training procedures if needed

    def use(self, state):
        # Use the Q-network to predict Q-values for all actions
        q_values = [self.q_value(state, action) for action in self.env.valid_action_values]
        return q_values  # Returns Q-values for all possible actions
    
   




class Experiment:
    def __init__(self, environment, agent):
        # Initialize the environment and agent for the experiment
        self.env = environment
        self.agent = agent

    def train(self, parms, verbose=True):
        n_batches = parms['n_batches']
        n_steps_per_batch = parms['n_steps_per_batch']
        n_epochs = parms['n_epochs']
        method = parms['method']
        learning_rate = parms['learning_rate']
        epsilon = parms['initial_epsilon']
        final_epsilon = parms['final_epsilon']
        gamma = parms['gamma']

        epsilon_decay = np.exp((np.log(final_epsilon) - np.log(epsilon)) / n_batches)
        
        epsilon_trace = []
        outcomes = []
        all_rewards = []  # New list to accumulate rewards

        for batch in range(n_batches):
            self.agent.clear_samples()
            self.env.initialize()

            sum_rs = 0  # Track cumulative reward for the batch

            for step in range(n_steps_per_batch):
                obs = self.env.observe()
                action = self.agent.epsilon_greedy(obs, epsilon)

                self.env.act(action)
                r = self.env.reinforcement()
                sum_rs += r

                done = step == n_steps_per_batch - 1
                self.agent.add_sample(obs, action, r, done)

            avg_reward = sum_rs / n_steps_per_batch
            all_rewards.append(avg_reward)  # Store the average reward for the batch
            outcomes.append(avg_reward)

            self.agent.train(n_epochs, method, learning_rate, gamma)

            epsilon_trace.append(epsilon)
            epsilon *= epsilon_decay

            # Print the mean of all rewards so far if verbose is True
            if verbose and (len(outcomes) % (n_batches // 20) == 0):
                mean_reward_so_far = np.mean(all_rewards)
                print(f'{len(outcomes)}/{n_batches} batches, Mean reward so far: {mean_reward_so_far:.4f}')

        return outcomes

    def test(self, n_trials, n_steps, epsilon=0.0, graphics=True):
        sum_rs = 0
        for trial in range(n_trials):
            self.env.initialize()
            points = np.zeros((n_steps, 2))  # Track cart position and pole angle
            actions = np.zeros((n_steps))
            Q_values = np.zeros((n_steps))

            for i in range(n_steps):
                obs = self.env.observe()
                action = self.agent.epsilon_greedy(obs, epsilon)
                Q = self.agent.q_value(obs, action)
                self.env.act(action)
                sum_rs += self.env.reinforcement()

                # Record the cart position and pole angle
                points[i] = obs[:2]  # Store cart position and pole angle
                actions[i] = action
                Q_values[i] = Q

            if graphics:
                plt.plot(points[:, 0], label='Cart Position')
                plt.plot(points[:, 1], label='Pole Angle')
                plt.xlabel('Steps')
                plt.legend()
                plt.show()

        return sum_rs / (n_trials * n_steps)

if __name__ == '__main__':
    # Test just the initial position
    print("Testing initial cart-pole position...")
    cart = CartPole()
    cart.cartpole.initDisplay()
    
    try:
        for _ in range(100):
            cart.cartpole.draw()
            time.sleep(1/30)
    except Exception as e:
        print(f"Error during visualization: {e}")
        
    print("\nStarting full training...")
    
    # Create environment and agent
    cartpole_env = CartPole()
    agent = QnetAgent(cartpole_env, [128, 64], 'max')
    experiment = Experiment(cartpole_env, agent)

    # Define training parameters
    parms = {
        'n_batches': 200,
        'n_steps_per_batch': 100,
        'n_epochs': 5,
        'method': 'sgd',
        'learning_rate': 0.01,
        'initial_epsilon': 1.0,
        'final_epsilon': 0.05,
        'gamma': 0.99
    }

    # Train and visualize
    outcomes, epsilon_trace = experiment.train(parms)
    
    print("\nTraining complete. Starting animation...")
    mean_reward = experiment.animate(1000)
    print(f"\nMean reward during animation: {mean_reward:.3f}")


 