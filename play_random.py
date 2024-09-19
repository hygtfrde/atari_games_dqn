import gym

# Create the SpaceInvaders environment
env = gym.make('SpaceInvaders-v0')

# Reset the environment to get the initial state
state = env.reset()

# Play 5 episodes
for _ in range(5):
    done = False
    while not done:
        env.render(mode='human')
        # Take a random action
        action = env.action_space.sample()
        # Step the environment
        state, reward, done, info = env.step(action)

env.close()