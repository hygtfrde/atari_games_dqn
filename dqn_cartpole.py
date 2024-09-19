import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os
import datetime

output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hyperparameters
episodes = 5
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

# Function to run a trained agent and visualize performance
def test_agent(agent, env, num_episodes=5):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, state_size])
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode + 1}/{num_episodes}, Reward: {total_reward}")

    return total_rewards


# Initialize agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training the agent
start_time = time.time()

# Prompt user to decide whether to display the game window
show_game = input("Do you want to display the game while training (Y/N)? ").strip().lower()

for episode in range(episodes):
    state, _ = env.reset()  # Ensure reset happens before render
    if show_game == 'y':
        env.render()  # Render the game only if the user wants to see it

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
        
end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"Training took {int(minutes)} minutes and {int(seconds)} seconds.")


# Test the trained agent
total_rewards = test_agent(agent, env)

# Generate the timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Plot the performance
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title(f'Performance of the Trained Pac-Man Agent - {timestamp}')

# Save the plot to the 'visualizations' directory
plot_filename = os.path.join(output_dir, f'pacman_performance_{timestamp}.png')
plt.savefig(plot_filename)

# Optionally show the plot for a few seconds and then close it
plt.show(block=False)
plt.pause(3)
plt.close()

print(f"Plot saved to {plot_filename}")