import numpy as np
import matplotlib.pyplot as plt
from math import pi
import time
import neuralnetworksA4 as nn
import rl_framework as rl
import cartpole_play as cp
import Box2D.b2 as b2

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
        """Initialize cart-pole with pole upright"""
        # Create new CartPole instance
        self.cartpole = cp.CartPole()
        # Initialize cartpole's action
        self.cartpole.action = 0
        
        # Set cart position to center
        self.cartpole.cart.position = (0.0, self.cartpole.cart.position[1])
        
        # Set cart velocity to zero
        self.cartpole.cart.linearVelocity = (0.0, 0.0)
        
        # Set pole position to upright (pi radians from default position)
        self.cartpole.pole.angle = pi  # This makes it point up
        
        # Set pole angular velocity to zero
        self.cartpole.pole.angularVelocity = 0.0
        
        # Force Box2D to update the physics
        for _ in range(5):  # Multiple steps to ensure stability
            self.cartpole.world.Step(1.0/30.0, 6, 2)
            self.cartpole.world.ClearForces()

        # Set initial state
        self.state = self.observe()

    def act(self, action):
        """Apply action and get new state"""
        # Store the action for visualization
        self.cartpole.action = action
        # Apply the action
        self.cartpole.act(action)
        self.state = self.observe()

    def valid_actions(self):
        return self.valid_action_values

    def observe(self):
        """Get the current state observation"""
        x, xdot, angle, angledot = self.cartpole.sense()
        # Normalize angle to be between -pi and pi, with 0 being upright
        angle = angle - pi  # Shift by pi so upright is 0
        angle = ((angle + pi) % (2 * pi)) - pi  # Normalize to [-pi, pi]
        self.state = np.array([x, xdot, angle, angledot])
        return self.state

    def act(self, action):
        """Apply action and get new state"""
        self.cartpole.act(action)
        self.state = self.observe()

    def reinforcement(self):
        """Reward function"""
        x, xdot, angle, angledot = self.state
        
        # Center angle at zero (upright position)
        angle = abs(angle)  # Now zero means upright
        
        # Reward based on angle
        if angle < pi/6:  # Within 30 degrees
            angle_reward = 1.0
        elif angle < pi/2:  # Within 90 degrees
            angle_reward = 0.5
        else:
            angle_reward = -1.0
            
        # Penalty for being far from center
        position_penalty = -abs(x/2.4)  # Normalized by track half-width
        
        # Combined reward
        reward = angle_reward + 0.2 * position_penalty
        
        # Terminal failure conditions
        if abs(x) > 2.4 or angle > pi/2:
            reward = -2.0
            
        return reward

    def terminal_state(self, state=None):
        """Check if current state is terminal"""
        if state is None:
            state = self.state
        x, xdot, angle, angledot = state
        return abs(x) > 2.4 or abs(angle) > pi/2

    def __str__(self):
        if self.state is None:
            return "CartPole: uninitialized"
        x, xdot, angle, angledot = self.state
        return f"CartPole: x={x:.2f}, xdot={xdot:.2f}, angle={angle:.2f}, angledot={angledot:.2f}"

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

    #extra function
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
        """Calculate Q values for next states"""
        if len(self.X) == 0:
            return np.array([])

        self.X = np.vstack(self.X)
        self.R = np.array(self.R).reshape(-1, 1)
        self.Done = np.array(self.Done).reshape(-1, 1)

        Qn = np.zeros_like(self.R)
        last_steps = np.where(self.Done)[0]

        if len(last_steps) == 0:
            self.Done[-1] = True
            last_steps = np.array([len(self.Done) - 1])

        first = 0
        for last_step in last_steps:
            if last_step > first:  # Only process if there are states between first and last_step
                next_states = self.X[first + 1:last_step + 1]
                if len(next_states) > 0:  # Only calculate Q values if there are next states
                    Qn[first:last_step] = self.use(next_states)
            first = last_step + 1

        return Qn

    def train(self, n_epochs, method, learning_rate, gamma, verbose=True):
        """Train the Q-network"""
        if len(self.X) == 0:
            return

        Qn = self.update_Qn()
        
        # Make sure all arrays are properly shaped
        if len(Qn) > 0:
            T = self.R_sign * self.R + gamma * Qn
            non_terminal = ~self.Done.flatten()
            
            if np.any(non_terminal):
                X_train = self.X[non_terminal]
                T_train = T[non_terminal]

                self.Qnet.train(X_train, T_train, X_train, T_train,
                               n_epochs=n_epochs, method=method,
                               learning_rate=learning_rate,
                               verbose=verbose)

class Experiment:
    def __init__(self, environment, agent):
        self.env = environment
        self.agent = agent
        self.env.initialize()
        self.agent.initialize()
        self.best_weights = None
        self.best_reward = float('-inf')

    def train(self, parms, verbose=True):
        n_batches = parms['n_batches']
        n_steps_per_batch = parms['n_steps_per_batch']
        n_epochs = parms['n_epochs']
        method = parms['method']
        learning_rate = parms['learning_rate']
        final_epsilon = parms['final_epsilon']
        epsilon = parms['initial_epsilon']
        gamma = parms['gamma']

        epsilon_decay = np.exp(np.log(final_epsilon/epsilon) / n_batches)
        epsilon_trace = []
        outcomes = []

        for batch in range(n_batches):
            self.agent.clear_samples()
            batch_rewards = []

            # Run multiple episodes per batch
            n_episodes = 5
            for _ in range(n_episodes):
                self.env.initialize()
                episode_rewards = []

                for step in range(n_steps_per_batch):
                    obs = self.env.observe()
                    action = self.agent.epsilon_greedy(epsilon)
                    
                    self.env.act(action)
                    r = self.env.reinforcement()
                    done = self.env.terminal_state()
                    
                    self.agent.add_sample(obs, action, r, done)
                    episode_rewards.append(r)
                    
                    if done:
                        break

                batch_rewards.extend(episode_rewards)

            mean_reward = np.mean(batch_rewards)
            outcomes.append(mean_reward)

            # Train on collected samples
            self.agent.train(n_epochs, method, learning_rate, gamma)

            # Update best weights if performance improved
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.best_weights = self.agent.Qnet.all_weights.copy()

            epsilon *= epsilon_decay
            epsilon_trace.append(epsilon)

            if verbose and (batch % (n_batches // 20) == 0 or batch == n_batches - 1):
                print(f'Batch {batch}, Mean reward: {mean_reward:.3f}, '
                      f'Best: {self.best_reward:.3f}, Epsilon: {epsilon:.3f}')

        # Restore best weights
        if self.best_weights is not None:
            self.agent.Qnet.all_weights[:] = self.best_weights

        return outcomes, epsilon_trace

    def animate(self, n_steps):
        self.env.initialize()
        self.env.cartpole.initDisplay()
        rewards = []

        for _ in range(n_steps):
            obs = self.env.observe()
            action = self.agent.epsilon_greedy(0.0)
            self.env.act(action)
            r = self.env.reinforcement()
            rewards.append(r)
            self.env.cartpole.draw()
            time.sleep(1/30)

        return np.mean(rewards)

    # Add to the Experiment class:

    def test(self, n_steps):
        """Test agent performance from different starting angles."""
        states_actions = []
        sum_r = 0.0
        for initial_angle in [0, pi/2.0, -pi/2.0, pi]:
            self.env.cartpole.cart.position = (0.0, self.env.cartpole.cart.position[1])
            self.env.cartpole.cart.linearVelocity = (0.0, 0.0)
            self.env.cartpole.pole.angle = initial_angle
            self.env.cartpole.pole.angularVelocity = 0.0
            
            # Force physics update
            for _ in range(5):
                self.env.cartpole.world.Step(1.0/30.0, 6, 2)
                self.env.cartpole.world.ClearForces()
            
            for step in range(n_steps):
                obs = self.env.observe()
                action = self.agent.epsilon_greedy(epsilon=0.0)
                states_actions.append([*obs, action])
                self.env.act(action)
                r = self.env.reinforcement()
                sum_r += r
                
                # Break if terminal state reached
                if self.env.terminal_state():
                    break
                    
        return sum_r / (n_steps * 4), np.array(states_actions)
    
    def train(self, parms, verbose=True):
        """Modified train method to include testing."""
        n_batches = parms['n_batches']
        n_steps_per_batch = parms['n_steps_per_batch']
        n_epochs = parms['n_epochs']
        method = parms['method']
        learning_rate = parms['learning_rate']
        final_epsilon = parms['final_epsilon']
        epsilon = parms['initial_epsilon']
        gamma = parms['gamma']
    
        start_time = time.time()
        
        epsilon_decay = np.exp(np.log(final_epsilon/epsilon) / n_batches)
        epsilon_trace = []
        outcomes = []
        test_rewards = []
    
        for batch in range(n_batches):
            self.agent.clear_samples()
            batch_rewards = []
    
            # Run multiple episodes per batch
            n_episodes = 5
            for _ in range(n_episodes):
                self.env.initialize()
                episode_rewards = []
    
                for step in range(n_steps_per_batch):
                    obs = self.env.observe()
                    action = self.agent.epsilon_greedy(epsilon)
                    
                    self.env.act(action)
                    r = self.env.reinforcement()
                    done = self.env.terminal_state()
                    
                    self.agent.add_sample(obs, action, r, done)
                    episode_rewards.append(r)
                    
                    if done:
                        break
    
                batch_rewards.extend(episode_rewards)
    
            mean_reward = np.mean(batch_rewards)
            outcomes.append(mean_reward)
    
            # Train on collected samples
            self.agent.train(n_epochs, method, learning_rate, gamma)
    
            # Test current performance
            test_reward, _ = self.test(n_steps_per_batch)
            test_rewards.append(test_reward)
    
            epsilon *= epsilon_decay
            epsilon_trace.append(epsilon)
    
            if verbose and (batch % (n_batches // 20) == 0 or batch == n_batches - 1):
                print(f'Batch {batch}, Train reward: {mean_reward:.3f}, '
                      f'Test reward: {test_reward:.3f}, '
                      f'Mean test reward: {np.mean(test_rewards):.3f}, '
                      f'Epsilon: {epsilon:.3f}')
    
        execution_time = (time.time() - start_time) / 60  # Convert to minutes
        
        # Store performance metrics
        self.performance_stats = {
            'hidden_layers': self.agent.n_hiddens_each_layer,
            'n_batches': n_batches,
            'n_steps': n_steps_per_batch,
            'n_epochs': n_epochs,
            'initial_epsilon': parms['initial_epsilon'],
            'mean_test_reward': np.mean(test_rewards),
            'execution_minutes': execution_time
        }
        
        return outcomes, epsilon_trace, test_rewards
    
    def print_performance_stats(self):
        """Print performance statistics in tabular format."""
        stats = self.performance_stats
        print("\nPerformance Statistics:")
        print(f"{'Parameter':<15} {'Value':<10}")
        print("-" * 25)
        print(f"{'Hidden Layers':<15} {stats['hidden_layers']}")
        print(f"{'Batches':<15} {stats['n_batches']}")
        print(f"{'Steps/Batch':<15} {stats['n_steps']}")
        print(f"{'Epochs':<15} {stats['n_epochs']}")
        print(f"{'Init Epsilon':<15} {stats['initial_epsilon']:.2f}")
        print(f"{'Test Reward':<15} {stats['mean_test_reward']:.4f}")
        print(f"{'Exec Time (min)':<15} {stats['execution_minutes']:.2f}")
    
    def plot_test_run(self, n_steps=200):
        """Plot states from a test run."""
        _, states_actions = self.test(n_steps)
        
        plt.figure(figsize=(15, 10))
        
        # Plot angles
        plt.subplot(2, 2, 1)
        plt.plot(states_actions[:, 2])
        plt.title('Pole Angle vs Time')
        plt.xlabel('Step')
        plt.ylabel('Angle (radians)')
        
        # Plot positions
        plt.subplot(2, 2, 2)
        plt.plot(states_actions[:, 0])
        plt.title('Cart Position vs Time')
        plt.xlabel('Step')
        plt.ylabel('Position')
        
        # Plot actions
        plt.subplot(2, 2, 3)
        plt.plot(states_actions[:, -1])
        plt.title('Actions vs Time')
        plt.xlabel('Step')
        plt.ylabel('Action')
        
        # Plot phase space (angle vs angular velocity)
        plt.subplot(2, 2, 4)
        plt.plot(states_actions[:, 2], states_actions[:, 3])
        plt.title('Phase Space')
        plt.xlabel('Angle')
        plt.ylabel('Angular Velocity')
        
        plt.tight_layout()
        plt.show()

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