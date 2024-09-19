import gym
from gym.utils.play import play
import time

# Create the SpaceInvaders environment
env = gym.make('SpaceInvaders-v0', render_mode='human')

# Reset the environment to get the initial state
state, _ = env.reset()

# Play 5 episodes
for episode in range(5):
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Render the game
        env.render()
        
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Check if the episode is done
        done = terminated or truncated
        
        # Add a small delay to make the game visible
        time.sleep(0.01)
    
    print(f"Episode {episode + 1} finished. Total reward: {total_reward}. Steps: {steps}")

env.close()