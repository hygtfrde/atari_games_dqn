# Playing Atari Games with Gym

## Installation

1. **Install Gym with Atari support**:
    ```bash
    pip install gym[atari] gym[accept-rom-license]
    ```

2. **Install additional dependencies if needed**:
    ```bash
    pip install box2d-py
    pip install atari-py
    ```

## Running an Atari Game

1. **Create a Python script `play_atari.py` to run the game**:

    ```python
    import gym

    # Create the SpaceInvaders environment
    env = gym.make('SpaceInvaders-v0')

    # Reset the environment to get the initial state
    state = env.reset()

    # Play 5 episodes
    for _ in range(5):
        done = False
        while not done:
            env.render()
            # Take a random action
            action = env.action_space.sample()
            # Step the environment
            state, reward, done, info = env.step(action)

    env.close()
    ```

2. **Run the script**:
    ```bash
    python play_atari.py
    ```

## Troubleshooting

- **Rendering Issues**: If the game window doesn’t open, check your graphical settings or install additional libraries if necessary.
