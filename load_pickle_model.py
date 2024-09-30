import pickle

# Load the DQN model from the file
with open('dqn_agent_cnn.pkl', 'rb') as f:
    agent_cnn_loaded = pickle.load(f)
