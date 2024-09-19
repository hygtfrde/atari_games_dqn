import gym
from gym.utils.play import play

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

def main():
    game_envs = {
        1: "CartPole-v1",
        2: "SpaceInvaders-v0",
        3: "MsPacman-v0"
    }
    
    while True:
        choice = get_game_choice()
        
        if choice is None:
            break
        
        env_id = game_envs[choice]
        
        # Create the chosen environment
        env = gym.make(env_id, render_mode='human')
        
        # Play the game using user inputs (keyboard control)
        play(env, zoom=3)  # zoom=3 is optional to make the game window larger
        
        env.close()

if __name__ == "__main__":
    main()
