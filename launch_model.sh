#!/bin/bash

while true; do
    # Prompt the user with the available models
    echo "Which model would you like to run?"
    echo "1: CartPole (dqn_cartpole.py)"
    echo "2: Pacman (dqn_pacman.py)"
    echo "3: Space Invaders (dqn_space_invaders.py)"
    echo "Press Q or q to quit."

    # Read user input
    read -p "Enter the number of your choice: " choice

    # Handle quitting the script
    if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
        echo "Exiting the program."
        exit 0
    fi

    # Run the corresponding model based on user input
    case $choice in
        1)
            echo "Running CartPole model..."
            python3 dqn_cartpole.py
            echo "Model training done."
            break
            ;;
        2)
            echo "Running Pacman model..."
            python3 dqn_pacman.py
            echo "Model training done."
            break
            ;;
        3)
            echo "Running Space Invaders model..."
            python3 dqn_space_invaders.py
            echo "Model training done."
            break
            ;;
        *)
            echo "Invalid input. Please enter a number from 1 to 3, or Q to quit."
            ;;
    esac
done
