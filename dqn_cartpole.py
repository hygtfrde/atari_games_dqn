import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque

# Hyperparameters
episodes = 50
max_steps = 100
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network to approximate Q-value function
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),  # Define the input shape
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mse')
        return model

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decrease exploration rate
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Initialize agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training the agent
for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state, dtype=np.float32)
        next_state = np.reshape(next_state, [1, state_size])

        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break
