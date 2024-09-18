import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2

# Hyperparameters
episodes = 5  # More episodes for Pac-Man
max_steps = 100
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000
train_every = 4

# Initialize the MsPacman environment
env = gym.make('MsPacman-v0')

def preprocess_state(state):
    """Preprocesses the game screen state for input into the DQN."""
    # Ensure state is a numpy array
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    # Check if state is already 2D (grayscale)
    if len(state.shape) == 2:
        resized = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    # If state is 3D (color image)
    elif len(state.shape) == 3:
        resized = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected state shape: {state.shape}")
    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0
    return normalized.flatten()  # Flatten the state to a 1D array for the network

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        """Initialize the DQN agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  # Memory buffer
        self.epsilon = epsilon  # Exploration rate
        self.model = self._build_model()  # Main model
        self.target_model = self._build_model()  # Target model
        self.update_target_model()  # Sync weights between models

    def _build_model(self):
        """Builds a neural network model for the Q-value function approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Output Q-values for each action
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience tuple in memory."""
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        state = np.array(state).reshape(-1, self.state_size)  # Ensure correct input shape
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit: take action with highest Q-value

    def train(self):
        """Trains the agent by sampling experiences from memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([s[0] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])
        
        q_values = self.model.predict(states)
        q_next_values = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + gamma * np.amax(q_next_values[i])
            q_values[i][action] = target
        
        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay  # Decay epsilon after each step

    def update_target_model(self):
        """Copies the weights from the model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

# Initialize the DQN agent
state_size = 84 * 84  # Flattened 84x84 image (same as for Space Invaders)
action_size = env.action_space.n  # Number of possible actions in Pac-Man
agent = DQNAgent(state_size, action_size)

# Training the agent
for episode in range(episodes):
    # state = env.reset()  # Get initial state
    state, *_ = env.reset()  # Unpack only the first element (state)
    print(f"Initial state type: {type(state)}, shape: {np.array(state).shape}")
    
    state = env.reset()[0]  # Extract the state from the returned tuple
    state = preprocess_state(state)
    state = np.reshape(state, (1, state_size))  # Add batch dimension
        
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)  # Select action
        next_state, reward, done, _, info = env.step(action)
        next_state = preprocess_state(next_state)
        next_state = np.reshape(next_state, (state_size,))  # Reshape next state

        agent.store_experience(state, action, reward, next_state, done)  # Store experience

        if step % train_every == 0:
            agent.train()  # Train agent periodically

        state = next_state  # Move to the next state
        total_reward += reward  # Accumulate rewards

        if done:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

    if episode % 10 == 0:
        agent.update_target_model()  # Update target model periodically

env.close()

# Function to run a trained agent and visualize performance
def test_agent(agent, env, num_episodes=5):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()  # Unpack only the first element (state)
        state = preprocess_state(state)
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state)  # Choose action
            next_state, reward, done, truncated, _ = env.step(action)  # Take action
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, (1, state_size))
            total_reward += reward
            state = next_state  # Move to the next state

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode + 1}/{num_episodes}, Reward: {total_reward}")

    return total_rewards

# Test the trained agent
total_rewards = test_agent(agent, env)

# Plot the performance
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Trained Pac-Man Agent')
plt.show()