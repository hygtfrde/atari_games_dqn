import gym
import numpy as np
from gym.utils.play import play
import pygame

def get_game_choice():
    while True:
        print("Choose a game to play by entering the corresponding number:")
        print("1: CartPole")
        print("2: Space Invaders")
        print("3: Pacman")
        print("Press 'Q' or 'q' to quit.")
        
        choice = input().strip().lower()
        
        if choice in ['q', 'quit']:
            print("Exiting the program.")
            return None
        
        if choice in ['1', '2', '3']:
            return int(choice)
        
        print("Invalid input. Please enter a number from 1 to 3, or 'Q' to quit.")

def get_keys_to_action(env_id):
    if env_id == "CartPole-v1":
        return {
            (pygame.K_LEFT,): 0,  # Push cart to the left
            (pygame.K_RIGHT,): 1,  # Push cart to the right
        }
    elif env_id == "SpaceInvaders-v4":
        return None  # Use default key mapping
    elif env_id == "MsPacman-v4":
        return None  # Use default key mapping
    else:
        raise ValueError(f"No key mapping defined for {env_id}")

def play_game(env, keys_to_action=None):
    obs = env.reset()
    done = False
    cumulative_reward = 0

    while not done:
        env.render()
        action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                elif keys_to_action and event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    action = keys_to_action.get((event.key,), None)

        if action is None and keys_to_action:
            action = 0  # Default action

        if action is not None:
            obs, reward, done, _, _ = env.step(action)
            cumulative_reward += reward

    print(f"Game over! Cumulative reward: {cumulative_reward}")
    env.close()

def main():
    game_envs = {
        1: "CartPole-v1",
        2: "SpaceInvaders-v4",
        3: "MsPacman-v4"
    }
    
    pygame.init()
    
    while True:
        choice = get_game_choice()
        
        if choice is None:
            break
        
        env_id = game_envs[choice]
        
        # Create the chosen environment
        env = gym.make(env_id, render_mode='human')
        
        # Get the appropriate key-to-action mapping
        keys_to_action = get_keys_to_action(env_id)
        
        # Play the game using user inputs (keyboard control)
        play_game(env, keys_to_action)
        
    pygame.quit()

if __name__ == "__main__":
    main()