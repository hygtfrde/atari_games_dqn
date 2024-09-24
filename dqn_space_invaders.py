import gym
from gym.wrappers import OrderEnforcing
import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time

# Hyperparameters
episodes = 3
max_steps = 100
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000
train_every = 4

# Initialize the Space Invaders environment
env = gym.make('SpaceInvaders-v0', render_mode='human')
env = OrderEnforcing(env, disable_render_order_enforcing=True)


def preprocess_state(state):
    resized = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    return normalized.flatten()

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, model_type='cnn'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        
        # Select the model type
        self.model_type = model_type
        if self.model_type == 'cnn':
            self.model = self._build_model_cnn()
            self.target_model = self._build_model_cnn()
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()

        self.update_target_model()

    # 'Regular' Fully Connected Dense Model
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
    
    # CNN Model
    def _build_model_cnn(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(84, 84, 1)),  # 84x84 grayscale image input
            tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mse')
        return model

    def store_experience(self, state, action, reward, next_state, done):
        if self.model_type == 'cnn':
            # Store state without flattening for CNN
            self.memory.append((state, action, reward, next_state, done))
        else:
            # Flatten state for dense model
            self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape input based on model type
        if self.model_type == 'cnn':
            state = state.reshape(-1, 84, 84, 1)  # For CNN input
        else:
            state = state.reshape(-1, self.state_size)  # For dense model input

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        if self.model_type == 'cnn':
            # Reshape states for CNN input
            states = np.array([s[0] for s in minibatch]).reshape(batch_size, 84, 84, 1)
            next_states = np.array([s[3] for s in minibatch]).reshape(batch_size, 84, 84, 1)
        else:
            # Flatten states for dense model input
            states = np.array([s[0] for s in minibatch]).reshape(batch_size, self.state_size)
            next_states = np.array([s[3] for s in minibatch]).reshape(batch_size, self.state_size)

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

# For CNN model
agent_cnn = DQNAgent(state_size=84 * 84, action_size=env.action_space.n, model_type='cnn')

# For regular dense model
# agent_dense = DQNAgent(state_size=84 * 84, action_size=env.action_space.n, model_type='dense')




# Training the agent
start_time = time.time()

# Prompt user to decide whether to display the game window
show_game = input("Do you want to display the game while training (Y/N)? ").strip().lower() == 'y'

for episode in range(episodes):
    if show_game:
        env.reset()  # Reset the environment first
        env.render()  # Render the game only if the user wants to see it
 

for episode in range(episodes):
    state, _ = env.reset()  # Reset the environment and get the initial state
    state = preprocess_state(state)
    state = np.reshape(state, (1, state_size))  # Add batch dimension
        
    total_reward = 0

    for step in range(max_steps):
        if show_game:
            env.render()  # Render the game only if the user wants to see it

        action = agent_cnn.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = preprocess_state(next_state)
        next_state = np.reshape(next_state, (1, state_size))  # Add batch dimension

        agent_cnn.store_experience(state, action, reward, next_state, done)
        
        if step % train_every == 0:
            agent_cnn.train()

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent_cnn.epsilon:.2f}")
            break

    if episode % 10 == 0:
        agent_cnn.update_target_model()

env.close()

end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"Training took {int(minutes)} minutes and {int(seconds)} seconds.")

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
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, (1, state_size))
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode + 1}/{num_episodes}, Reward: {total_reward}")

    return total_rewards

# Test the trained agent
total_rewards = test_agent(agent_cnn, env)

# Print best episode score aka Final Score

# Plot the performance
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of the Trained Agent')
plt.show(block=False)
plt.pause(3)
plt.close()
