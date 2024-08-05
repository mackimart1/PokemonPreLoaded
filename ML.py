import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from explore import ExplorationEnvironment

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, text_embed_dim=768):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.text_embed_dim = text_embed_dim

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(state_dim + text_embed_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, state, text):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            text_embeddings = self.bert(**tokens).last_hidden_state[:, 0, :]
        combined = torch.cat((state_tensor.squeeze(0), text_embeddings), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)

    def get_action(self, state, text):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state), text)
                return torch.argmax(q_values).item()

    def update(self, state, text, action, reward, next_state, next_text, done):
        self.memory.append((state, text, action, reward, next_state, next_text, done))
        
        if len(self.memory) < 32:
            return 0

        batch = random.sample(self.memory, 32)
        states, texts, actions, rewards, next_states, next_texts, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states, texts).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states, next_texts).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.memory = checkpoint['memory']
                print("Model loaded successfully.")
            except:
                print("Error loading model. Starting with a fresh model.")
        else:
            print("No saved model found. Starting with a fresh model.")

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
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        
        new_x = max(0, min(self.width - 1, self.state['x'] + dx))
        new_y = max(0, min(self.height - 1, self.state['y'] + dy))
        
        self.state['x'] = new_x
        self.state['y'] = new_y
        
        reward = 0
        if self.grid[new_y][new_x] == 0:
            self.grid[new_y][new_x] = 1
            self.state['discovered_cells'] += 1
            reward += 10  # Reward for discovering a new cell

        self.state['visited_count'] += 1
        self.state['score'] += random.randint(0, 5)
        self.state['normalized_health'] = max(0, min(1, self.state['normalized_health'] + random.uniform(-0.05, 0.05)))
        self.state['enemy_count'] = random.randint(0, 3)
        self.state['danger_level'] = random.randint(0, 3)
        self.state['average_intensity'] = random.uniform(0, 100)
        self.state['enemies_defeated'] += random.randint(0, 1)
        self.state['low_health_flag'] = 1 if self.state['normalized_health'] < 0.3 else 0
        self.state['high_score_flag'] = 1 if self.state['score'] > 500 else 0

        reward += self.state['score'] / 100 + self.state['normalized_health'] - self.state['danger_level'] / 10

        done = self.state['visited_count'] >= 200 or self.state['discovered_cells'] == self.width * self.height or self.state['normalized_health'] <= 0

        return self.state, reward, done

def preprocess_state(state):
    return [
        state['normalized_health'],
        state['score'],
        state['enemies_defeated'],
        state['danger_level'],
        state['x'],
        state['y'],
        state['enemy_count'],
        state['low_health_flag'],
        state['high_score_flag'],
        state['average_intensity'],
        state['visited_count'],
        state['discovered_cells']
    ]

def generate_text_description(state):
    return f"Player at ({state['x']}, {state['y']}) with health {state['normalized_health']:.2f}. " \
           f"Score: {state['score']}. Discovered cells: {state['discovered_cells']}. " \
           f"Enemies defeated: {state['enemies_defeated']}. Danger level: {state['danger_level']}. " \
           f"Enemy count: {state['enemy_count']}. Low health: {'Yes' if state['low_health_flag'] else 'No'}. " \
           f"High score: {'Yes' if state['high_score_flag'] else 'No'}. " \
           f"Average intensity: {state['average_intensity']:.2f}."

def main():
    state_dim = 12
    action_dim = 4
    agent = Agent(state_dim, action_dim)
    
    model_path = "trained_model.pth"
    agent.load_model(model_path)

    writer = SummaryWriter('logs')

    num_episodes = 1000
    save_interval = 100
    env = ExplorationEnvironment()

    for episode in range(num_episodes):
        state = env.reset()
        processed_state = preprocess_state(state)
        text = generate_text_description(state)
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0
        
        while not done:
            # Assuming 'state' is your current game state
            processed_state = preprocess_state(state)
            text = generate_text_description(state)
            action = agent.get_action(processed_state, text)
            next_state, reward, done = env.step(action)
            processed_next_state = preprocess_state(next_state)
            next_text = generate_text_description(next_state)

            loss = agent.update(processed_state, text, action, reward, processed_next_state, next_text, done)
            episode_loss += loss
            processed_state = processed_next_state
            text = next_text
            total_reward += reward
            steps += 1

        if episode % 10 == 0:
            agent.update_target_model()

        avg_loss = episode_loss / steps if steps > 0 else 0
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Average Loss', avg_loss, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Discovered Cells', state['discovered_cells'], episode)

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}, Discovered Cells: {state['discovered_cells']}")

        if (episode + 1) % save_interval == 0:
            agent.save_model(model_path)
            print(f"Model saved at episode {episode + 1}")

    writer.close()
    agent.save_model(model_path)
    print("Training completed. Final model saved.")

if __name__ == "__main__":
    main()