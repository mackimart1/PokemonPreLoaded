import subprocess
import time
import os
import torch
import numpy as np
import threading
import queue
import pyautogui
import cv2
from ML import Agent, preprocess_state
from keyboard import SimulatedKeyboard
from key_mapper import KeyMapper
from ScreenMirror import ScreenMirror, ComputerVision
import tkinter as tk
from gamestate import process_game_state
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)

# Path to the game executable
GAME_PATH = r"C:\Users\macki\OneDrive\Desktop\POKEReloaded\Proyecto Reloaded The Last Beta.exe"
GAME_DIR = os.path.dirname(GAME_PATH)


def user_input_thread(input_queue):
    while True:
        user_input = input("Enter command (start/stop/quit): ").lower()
        input_queue.put(user_input)
        if user_input == 'quit':
            break

def get_default_state():
    return {
        'normalized_health': 0.0,
        'score': 0,
        'enemies_defeated': 0,
        'danger_level': 0,
        'x_position': 0,
        'y_position': 0,
        'enemy_count': 0,
        'low_health_flag': 1,
        'high_score_flag': 0,
        'average_intensity': 0,
        'game_over': 0,
        'level_complete': 0
    }

def start_bot(mapped_keys=None):
    if not os.path.isfile(GAME_PATH):
        print(f"Error: Game executable not found at {GAME_PATH}")
        return

    try:
        process = subprocess.Popen(
            GAME_PATH,
            cwd=GAME_DIR,
            env=os.environ.copy(),
            shell=True
        )
        print(f"Game launched with PID: {process.pid}")
    except Exception as e:
        print(f"Error launching game: {e}")
        return

    print("Waiting for the game to start...")
    time.sleep(10)

    root = tk.Tk()
    cv = ComputerVision(root)
    keyboard = SimulatedKeyboard(mapped_keys)
    
    # Initialize the agent
    state_dim = 12  # Adjust this based on the number of items in get_default_state()
    action_dim = keyboard.get_key_count()
    agent = Agent(state_dim, action_dim)
    model_path = "game_knowledge.pth"
    
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            print("Loaded existing model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model.")
    else:
        print("No existing model found. Creating new model.")

    input_queue = queue.Queue()
    input_thread = threading.Thread(target=user_input_thread, args=(input_queue,))
    input_thread.start()

    print("Bot initialized. Type 'start' to activate, 'stop' to pause, and 'quit' to exit.")
    bot_active = False

    try:
        episode = 0
        max_episodes = 1000
        while episode < max_episodes:
            episode += 1
            print(f"Starting episode {episode}")
            
            step = 0
            max_steps = 1000
            total_reward = 0
            state = None
            prev_state = None
            action = None
            
            while step < max_steps:
                try:
                    user_command = input_queue.get_nowait()
                    if user_command == 'start':
                        bot_active = True
                        print("Bot activated.")
                    elif user_command == 'stop':
                        bot_active = False
                        print("Bot paused.")
                    elif user_command == 'quit':
                        print("Quitting...")
                        agent.save_model(model_path)
                        print("Model saved")
                        return
                except queue.Empty:
                    pass

                if not bot_active:
                    time.sleep(0.1)
                    continue

                try:
                    screen = cv.capture_screenshot()
                    new_state_dict = cv.extract_game_state(screen)
                    processed_state_dict = process_game_state(new_state_dict)
                    
                    # Use defaultdict to ensure all keys are present
                    default_state = defaultdict(float, get_default_state())
                    default_state.update(processed_state_dict)
                    new_state = dict(default_state)

                    if state is not None and action is not None:
                        reward = calculate_reward(new_state, prev_state or new_state)
                        total_reward += reward
                        agent.update(state, action, reward, new_state, check_episode_end(new_state))

                    prev_state = state
                    state = new_state
                    action = agent.get_action(state, "")
                    print(f"Step {step}: Selected action: {action}")
                    print(f"Current state: {state}")

                    root.update()

                    keyboard.press_key(action)
                    time.sleep(0.1)  # Allow time for action to take effect

                    if check_episode_end(new_state):
                        print(f"Episode {episode} ended after {step} steps. Total reward: {total_reward}")
                        break

                    step += 1
                except Exception as e:
                    print(f"Error in game loop: {e}")
                    print(f"Current state: {new_state_dict}")
                    print(f"Processed state: {processed_state_dict}")
                    continue

            if episode % 10 == 0:
                agent.save_model(model_path)
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    except KeyboardInterrupt:
        print("Bot stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        agent.save_model(model_path)
        print("Model saved")
        input_thread.join(timeout=1)
        try:
            process.terminate()
            print("Game closed.")
        except Exception as e:
            print(f"Error closing the game: {e}")
        root.destroy()


def calculate_reward(state, prev_state):
    reward = 0
    
    # Reward for health changes
    health_diff = state['normalized_health'] - prev_state['normalized_health']
    reward += health_diff * 100  # Increase reward for health preservation
    
    # Reward for score increase
    score_diff = state['score'] - prev_state['score']
    reward += score_diff
    
    # Reward for defeating enemies
    enemies_defeated_diff = state['enemies_defeated'] - prev_state['enemies_defeated']
    reward += enemies_defeated_diff * 50
    
    # Penalty for high danger level
    reward -= state['danger_level'] * 10
    
    # Time penalty
    reward -= 0.1
    
    # Bonus for completing a level
    if state.get('level_complete', 0) == 1:
        reward += 1000
    
    return np.clip(reward, -100, 100)  # Clip reward to prevent extreme values



def check_episode_end(state):
    # End episode if health is zero or game is over
    if state['normalized_health'] <= 0 or state.get('game_over', 0) == 1:
        return True
    
    # End episode if level is complete
    if state.get('level_complete', 0) == 1:
        return True
    
    return False


def process_game_state(state_dict):
    """
    Process the game state dictionary extracted from the screenshot.

    Args:
        state_dict (dict): Game state dictionary containing various game state information.

    Returns:
        dict: Processed game state dictionary with normalized values.
    """
    try:
        # Normalize health value to be between 0 and 1
        health = state_dict.get('health', 0) / 100

        # Extract score and enemies defeated
        score = state_dict.get('score', 0)
        enemies_defeated = state_dict.get('enemies_defeated', 0)

        # Calculate danger level based on proximity to enemies
        danger_level = calculate_danger_level(state_dict)

        # Create processed game state dictionary
        processed_state = {
            'normalized_health': health,
            'score': score,
            'enemies_defeated': enemies_defeated,
            'danger_level': danger_level,
            'x_position': state_dict.get('x_position', 0),
            'y_position': state_dict.get('y_position', 0),
            'enemy_count': state_dict.get('enemy_count', 0),
            'low_health_flag': int(health < 0.2),
            'high_score_flag': int(score > 1000)
        }

        return processed_state

    except KeyError as e:
        logger.error(f"KeyError in process_game_state: {e}")
        return {}
    except TypeError as e:
        logger.error(f"TypeError in process_game_state: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error in process_game_state: {e}")
        return {}

def calculate_danger_level(state_dict):
    """
    Calculate the danger level based on proximity to enemies.

    Args:
        state_dict (dict): Game state dictionary containing enemy positions.

    Returns:
        float: Danger level value between 0 and 1.
    """
    # Extract enemy positions
    enemy_positions = state_dict.get('enemy_positions', [])

    # Calculate distance to each enemy
    distances = [calculate_distance(state_dict['x_position'], state_dict['y_position'], enemy_x, enemy_y) for enemy_x, enemy_y in enemy_positions]

    # Calculate danger level based on minimum distance
    danger_level = 1 - min(distances) / max(distances) if distances else 0

    return danger_level

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1 (int): X-coordinate of point 1.
        y1 (int): Y-coordinate of point 1.
        x2 (int): X-coordinate of point 2.
        y2 (int): Y-coordinate of point 2.

    Returns:
        float: Euclidean distance between the two points.
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def main():
    print("Enter 'k' to start the bot, or 'm' to map custom keys.")
    
    while True:
        user_input = input().lower()
        if user_input == 'k':
            start_bot()
            break
        elif user_input == 'm':
            key_mapper = KeyMapper()
            mapped_keys = key_mapper.start_mapping()
            print("Mapped keys:", mapped_keys)
            start_bot(mapped_keys)
            break
        else:
            print("Invalid input. Please enter 'k' or 'm'.")

if __name__ == "__main__":
    main()



