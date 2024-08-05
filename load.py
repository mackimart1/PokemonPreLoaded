import torch
from ML import DQN  # Make sure this import works

def load_model(path, state_dim, action_dim):
    # Create a new instance of the DQN
    model = DQN(state_dim, action_dim)
    
    # Load the state dict
    state_dict = torch.load(path)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

# Usage
state_dim = 10  # Make sure this matches your original model
action_dim = 6  # Make sure this matches your original model
model_path = "observational_learned_model.pth"

loaded_model = load_model(model_path, state_dim, action_dim)

# Now loaded_model is your neural network, ready to use!