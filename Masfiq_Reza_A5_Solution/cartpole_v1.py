import numpy as np
from math import pi
import Box2D.b2 as b2
import cartpole_play as cp
import rl_framework as rl

import neuralnetworksA4 as nn

import matplotlib as plt 

class CartPole(rl.Environment):
    def __init__(self):
        self.cartpole = cp.CartPole()
        # Initialize cartpole's action
        self.cartpole.action = 0  # Initialize with no action
        
        self.valid_action_values = [-1, 0, 1]
        self.observation_size = 4  # x xdot a adot
        self.action_size = 1
        self.observation_means = [0, 0, 0, 0]
        self.observation_stds = [1, 1, 1, 1]
        self.action_means = [0.0]
        self.action_stds = [1.0]
        self.Q_means = [0]
        self.Q_stds = [1]
        self.state = None
        self.initialize()  # Initialize immediately
        
    def initialize(self):
        """Set up the cart-pole system in its initial balanced state."""
        # Recreate the CartPole instance to ensure fresh initialization
        self.cartpole = cp.CartPole()
        # Reset any applied force or action to zero
        self.cartpole.action = 0
    
        # Center the cart on the track and reset its velocity
        self.cartpole.cart.position = (0.0, self.cartpole.cart.position[1])
        self.cartpole.cart.linearVelocity = (0.0, 0.0)
        
        # Set the pole to be upright with no initial rotation or angular velocity
        self.cartpole.pole.angle = pi  # Upright position (rotated by pi radians)
        self.cartpole.pole.angularVelocity = 0.0
        
        # Run a few simulation steps to stabilize the setup
        for _ in range(5):  # Multiple steps ensure initial state stability
            self.cartpole.world.Step(1.0 / 30.0, 6, 2)
            self.cartpole.world.ClearForces()

        # Capture the initial observation
        self.state = self.observe()

    def act(self, action):
        """Apply a specified action to the cart-pole system and update the state."""
        # Store the action in the cartpole instance for visualization or logging
        self.cartpole.action = action
        # Apply the action within the simulation environment
        self.cartpole.act(action)
        # Update the current state observation after applying the action
        self.state = self.observe()
        
    def observe(self):
        """Retrieve the current observation from the cart-pole environment."""
        # Get the cart's position, velocity, pole angle, and angular velocity
        x, xdot, angle, angledot = self.cartpole.sense()

        # Adjust angle so that upright (pi) is represented as 0
        angle = angle - pi
        # Normalize angle within the range [-pi, pi]
        angle = ((angle + pi) % (2 * pi)) - pi

        # Create the state vector and update the current state
        self.state = np.array([x, xdot, angle, angledot])
        return self.state
    
    def valid_actions(self):
        """Return the list of valid actions available in the environment."""
        return self.valid_action_values
    
    def reinforcement(self):
        """Calculate the reward based on the current state."""
        x, xdot, angle, angledot = self.state
        
        # Center the angle around zero for the upright position
        angle = abs(angle)  # Upright angle will have the smallest value

        # Reward structure based on angle proximity to upright position
        if angle < pi / 6:  # Within 30 degrees
            angle_reward = 1.0
        elif angle < pi / 2:  # Within 90 degrees
            angle_reward = 0.5
        else:
            angle_reward = -1.0  # High penalty for large deviation

        # Penalize being far from the center of the track
        position_penalty = -abs(x / 2.4)  # Normalized by track half-width

        # Combine the angle reward and position penalty
        reward = angle_reward + 0.2 * position_penalty

        # Terminal condition with high penalty if the cart exceeds boundaries
        if abs(x) > 2.4 or angle > pi / 2:
            reward = -2.0  # Immediate high penalty for failure conditions

        return reward
    
    def terminal_state(self, state=None):
        """Check if the current state has reached a terminal condition."""
        # Use the current state if no specific state is provided
        if state is None:
            state = self.state
        
        # Unpack state variables
        x, xdot, angle, angledot = state

        # Check terminal conditions
        # The episode ends if the cart moves too far from the center or the pole angle exceeds 90 degrees
        return abs(x) > 2.4 or abs(angle) > pi / 2

    def __str__(self):
        """Provide a readable summary of the cart-pole's current state."""
        # Check if the state is initialized
        if self.state is None:
            return "CartPole: uninitialized"
        
        # Unpack state variables for display
        x, xdot, angle, angledot = self.state
        return (f"Cart Position: {x:.2f}, Velocity: {xdot:.2f}, "
                f"Pole Angle: {angle:.2f}, Angular Velocity: {angledot:.2f}")


class QnetAgent(rl.Agent):
    def initialize(self):
        env = self.env
        ni = env.observation_size + env.action_size
        self.Qnet = nn.NeuralNetwork(ni, self.n_hiddens_each_layer, 1)
        self.Qnet.X_means = np.array(env.observation_means + env.action_means)
        self.Qnet.X_stds = np.array(env.observation_stds + env.action_stds)
        self.Qnet.T_means = np.array(env.Q_means)
        self.Qnet.T_stds = np.array(env.Q_stds)
        self.clear_samples()
        
    def clear_samples(self):
        """Reset all sample storage arrays."""
        self.samples_X = []  # Observations and actions
        self.rewards = []    # Rewards received
        self.dones = []      # Flags indicating episode termination
        
    def add_sample(self, observation, action, reward, done):
        """Store a new experience sample."""
        # Combine observation and action into a single input sample
        sample = np.hstack((observation, action))
        
        # Append to sample storage
        self.samples_X.append(sample)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def use(self, X):
        """Calculate Q-value predictions for given inputs."""
        # Ensure input X is reshaped correctly if it's a single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Use the Q-network to predict Q-values
        return self.Qnet.use(X)

    def update_Qn(self):
        """Compute Q-values for the next states in the episode."""
        # If there are no samples, return an empty array
        if len(self.samples_X) == 0:
            return np.array([])

        # Convert lists to arrays for efficient calculations
        self.samples_X = np.vstack(self.samples_X)
        self.rewards = np.array(self.rewards).reshape(-1, 1)
        self.dones = np.array(self.dones).reshape(-1, 1)

        # Initialize Qn values with zeros
        Qn = np.zeros_like(self.rewards)
        end_indices = np.where(self.dones)[0]

        # Ensure there is at least one terminal state
        if len(end_indices) == 0:
            self.dones[-1] = True
            end_indices = np.array([len(self.dones) - 1])

        start = 0
        for end in end_indices:
            if end > start:
                next_states = self.samples_X[start + 1:end + 1]
                if len(next_states) > 0:
                    Qn[start:end] = self.use(next_states)
            start = end + 1

        return Qn
    
    def train(self, n_epochs, method, learning_rate, gamma, verbose=True):
        """Train the Q-network using stored samples and calculated targets."""
        # If there are no samples, skip training
        if len(self.samples_X) == 0:
            return

        # Calculate Q-values for the next states
        Qn = self.update_Qn()
        
        # If Qn is empty, no training can be performed
        if len(Qn) == 0:
            return
        
        # Target values for training, incorporating discount factor and future rewards
        targets = self.rewards + gamma * Qn
        non_terminal = ~self.dones.flatten()  # Only update for non-terminal states

        # Select samples where the state is not terminal
        if np.any(non_terminal):
            X_train = self.samples_X[non_terminal]
            T_train = targets[non_terminal]

            # Train the Q-network with the selected samples and targets
            self.Qnet.train(
                X_train, T_train, X_train, T_train,
                n_epochs=n_epochs,
                method=method,
                learning_rate=learning_rate,
                verbose=verbose
            )
            
            
class Experiment:           
            
    def __init__(self, environment, agent):

        self.env = environment
        self.agent = agent

        self.env.initialize()
        self.agent.initialize()
        # self.best_weights = None
        # self.best_reward = float('-inf')
        
    def train(self, parms, verbose=True):
        """Train the agent within the environment using specified parameters."""
        # Extract training parameters
        n_batches = parms['n_batches']
        n_steps_per_batch = parms['n_steps_per_batch']
        n_epochs = parms['n_epochs']
        method = parms['method']
        learning_rate = parms['learning_rate']
        epsilon = parms['initial_epsilon']
        final_epsilon = parms['final_epsilon']
        gamma = parms['gamma']

        # Calculate epsilon decay to gradually reduce exploration
        epsilon_decay = np.exp((np.log(final_epsilon) - np.log(epsilon)) / n_batches)

        # Track epsilon and outcomes for analysis
        epsilon_trace = []
        outcomes = []

        for batch in range(n_batches):
            self.agent.clear_samples()
            self.env.initialize()

            total_reward = 0

            for step in range(n_steps_per_batch):
                # Observe the current state and take an action
                observation = self.env.observe()
                action = self.agent.epsilon_greedy(epsilon)

                # Apply the action in the environment and receive the reward
                self.env.act(action)
                reward = self.env.reinforcement()
                total_reward += reward

                # Determine if the current state is terminal
                done = step == n_steps_per_batch - 1 or self.env.terminal_state()

                # Add the experience to the agent’s memory
                self.agent.add_sample(observation, action, reward, done)

                # Break out if the state is terminal
                if done:
                    break

            # Update the agent’s Q-network after each batch
            self.agent.train(n_epochs, method, learning_rate, gamma)

            # Track outcomes and epsilon decay
            outcomes.append(total_reward / n_steps_per_batch)
            epsilon_trace.append(epsilon)
            epsilon *= epsilon_decay

            # Display progress
            if verbose and (batch % max(1, n_batches // 10) == 0):
                print(f'Batch {batch + 1}/{n_batches}, Avg Reward: {np.mean(outcomes[-10:]):.2f}')
                
        # Restore best weights
        # if self.best_weights is not None:
        #     self.agent.Qnet.all_weights[:] = self.best_weights

        return outcomes, epsilon_trace
    
    
    def test(self, n_trials, n_steps, epsilon=0.0, graphics=True):
        """Evaluate the agent's performance over multiple trials."""
        total_reward = 0

        for trial in range(n_trials):
            # Reset environment for each trial
            self.env.initialize()
            trial_reward = 0

            for step in range(n_steps):
                # Observe the current state
                observation = self.env.observe()
                # Use the trained policy (exploit) with minimal exploration
                action = self.agent.epsilon_greedy(observation, epsilon)

                # Apply the action in the environment
                self.env.act(action)
                reward = self.env.reinforcement()
                trial_reward += reward

                # Check if the episode ends
                if self.env.terminal_state():
                    break

            # Track cumulative rewards over all trials
            total_reward += trial_reward

            # Optionally display graphics or logs
            if graphics:
                print(f"Trial {trial + 1}: Reward = {trial_reward}")

        # Calculate and return the average reward across trials
        avg_reward = total_reward / n_trials
        if graphics:
            print(f"\nAverage reward over {n_trials} trials: {avg_reward:.2f}")
        
        return avg_reward
    
    
    def animate(self, n_steps, epsilon=0.0):
        """Visualize a single episode of the agent interacting with the environment."""
        self.env.initialize()
        total_reward = 0

        for step in range(n_steps):
            # Observe the current state
            observation = self.env.observe()
            
            # Choose action using a nearly greedy policy (epsilon close to 0 for exploitation)
            action = self.agent.epsilon_greedy(epsilon)
            
            # Apply the action in the environment
            self.env.act(action)
            
            # Render the current state if visualization is supported
            if hasattr(self.env, "render"):
                self.env.render()

            # Accumulate the reward and check for terminal state
            total_reward += self.env.reinforcement()
            if self.env.terminal_state():
                break

        print(f"Total reward for animated episode: {total_reward}")
        return total_reward
    
    
    


    
    





