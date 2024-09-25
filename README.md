# Welcome to Atari Games
***

## Task
The task is to build an AI that can play three different Atari games: **CartPole**, **Space Invaders**, and **Pacman**. The challenge lies in training the AI to play these games using **Deep Q-Networks (DQN)**, a reinforcement learning technique that approximates the optimal action-value function via deep learning. Each game presents unique complexities, such as high-dimensional state spaces and varying reward structures, requiring different neural network architectures and hyperparameter tuning.

## Description
The problem is solved by implementing a **DQN (Deep Q-Network)** for each game. The AI learns to play the games by interacting with the game environment, storing its experiences in a replay buffer, and using those experiences to train a neural network to predict the best actions. The DQN model is initially built for the simpler CartPole game using fully connected layers and then adapted for Space Invaders and Pacman by using convolutional neural networks (CNNs) to handle high-dimensional inputs like game frames. Experience replay, target networks, and reward shaping are employed to stabilize training and improve performance.

## Installation
To set up the virtual env and install the project dependencies, follow these steps in the project repo:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
pip install gym[atari] gym[accept-rom-license]
```
```bash
# if Mac OS struggles with bracket notation for optional modules, instead use escape chars:
pip install gym\[atari\] gym\[accept-rom-license\]
# or try quoting:
pip install "gym[atari]" "gym[accept-rom-license]"
```


## Usage
The bash script `launch_model.sh` will prompt the user which model to train and test, then it will run. 
Run it with on Unix-like systems:
```
sh launch_model.sh
```
Alternatively, we can run a model to train and test individually.
To train the DQN model for CartPole, run the following command:
```
python CartPole/dqn_cartpole.py
```
To train the DQN model for Space Invaders, use the following:
```
python SpaceInvaders/dqn_space_invaders.py
```
And for Pacman:
```
python Pacman/dqn_pacman.py
```

For Hyperparameter tuning with Space Invaders:
```
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
```

### The Core Team


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt="Qwasar SV -- Software Engineering School's Logo" src="https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png" width='20px' /></span>
