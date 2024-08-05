# explore.py

import random
import numpy as np
import torch
from collections import deque

class ExplorationEnvironment:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.state = {
            'x': random.randint(0, self.width - 1),
            'y': random.randint(0, self.height - 1),
            'visited_count': 0,
            'discovered_cells': 0,
            'normalized_health': 1.0,
            'score': 0,
            'enemies_defeated': 0,
            'danger_level': 0,
            'enemy_count': 0,
            'low_health_flag': 0,
            'high_score_flag': 0,
            'average_intensity': 50.0
        }
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        return self.state

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        
        new_x = max(0, min(self.width - 1, self.state['x'] + dx))
        new_y = max(0, min(self.height - 1, self.state['y'] + dy))
        
        self.state['x'] = new_x
        self.state['y'] = new_y
        
        reward = 0
        if self.grid[new_y][new_x] == 0:
            self.grid[new_y][new_x] = 1
            self.state['discovered_cells'] += 1
            reward += 10  # Increased reward for discovering a new cell

        # Update other state variables
        self.state['visited_count'] += 1
        self.state['score'] += random.randint(0, 5)
        self.state['normalized_health'] = max(0, min(1, self.state['normalized_health'] + random.uniform(-0.05, 0.05)))
        self.state['enemy_count'] = random.randint(0, 3)
        self.state['danger_level'] = random.randint(0, 3)
        self.state['average_intensity'] = random.uniform(0, 100)
        self.state['enemies_defeated'] += random.randint(0, 1)
        self.state['low_health_flag'] = 1 if self.state['normalized_health'] < 0.3 else 0
        self.state['high_score_flag'] = 1 if self.state['score'] > 500 else 0

        # Additional reward components
        reward += self.state['score'] / 100 + self.state['normalized_health'] - self.state['danger_level'] / 10

        done = self.state['visited_count'] >= 200 or self.state['discovered_cells'] == self.width * self.height or self.state['normalized_health'] <= 0

        return self.state, reward, done

def preprocess_state(state, state_dim):
    default_state = {
        'normalized_health': 0.0,
        'score': 0,
        'enemies_defeated': 0,
        'danger_level': 0,
        'x_position': 0,
        'y_position': 0,
        'enemy_count': 0,
        'low_health_flag': 1,
        'high_score_flag': 0,
        'average_intensity': 0.0,
        'visited_count': 0,
        'discovered_cells': 0
    }
    default_state.update(state)
    state_values = list(default_state.values())
    if len(state_values) != state_dim:
        state_values = state_values[:state_dim] + [0] * max(0, state_dim - len(state_values))
    return dict(zip(default_state.keys(), state_values))

def generate_text_description(state):
    return f"Player at ({state['x']}, {state['y']}) with health {state['normalized_health']:.2f}. " \
           f"Score: {state['score']}. Discovered cells: {state['discovered_cells']}. " \
           f"Enemies defeated: {state['enemies_defeated']}. Danger level: {state['danger_level']}. " \
           f"Enemy count: {state['enemy_count']}. Low health: {'Yes' if state['low_health_flag'] else 'No'}. " \
           f"High score: {'Yes' if state['high_score_flag'] else 'No'}. " \
           f"Average intensity: {state['average_intensity']:.2f}."