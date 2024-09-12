import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2

# Hyperparameters
episodes = 50
max_steps = 1000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000
train_every = 4

# Initialize the Space Invaders environment
env = gym.make('SpaceInvaders-v0')

def preprocess_state(state):
    # Resize the image
    resized = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # Normalize
    normalized = gray / 255.0
    # Flatten
    return normalized.flatten()

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mse')
        return model

    # def store_experience(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))
        
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     q_values = self.model.predict(state)
    #     return np.argmax(q_values[0])
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :])
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        # states = np.array([s[0] for s in minibatch])
        # next_states = np.array([s[3] for s in minibatch])
        
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
            self.epsilon *= epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Initialize agent
state_size = 84 * 84  # Flattened 84x84 image
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training the agent
for episode in range(episodes):
    state = env.reset()[0]  # Get the state from the tuple
    
    
    # state = preprocess_state(state)
    # state = np.reshape(state, [1, state_size])
    
    state = preprocess_state(state)
    state = np.reshape(state, (state_size,))  # Not [1, state_size]
    
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # next_state = preprocess_state(next_state)
        # next_state = np.reshape(next_state, [1, state_size])
        
        next_state = preprocess_state(next_state)
        next_state = np.reshape(next_state, (state_size,))  # Not [1, state_size]

        agent.store_experience(state, action, reward, next_state, done)
        
        if step % train_every == 0:
            agent.train()

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

    if episode % 10 == 0:
        agent.update_target_model()

env.close()

# Function to run a trained agent and visualize the performance
def test_agent(agent, env, num_episodes=5):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()[0]  # Get the state from the tuple
        state = preprocess_state(state)
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode + 1}/{num_episodes}, Reward: {total_reward}")

    return total_rewards

# Test the trained agent
total_rewards = test_agent(agent, env)

# Plot the performance
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Trained Agent')
plt.show()
