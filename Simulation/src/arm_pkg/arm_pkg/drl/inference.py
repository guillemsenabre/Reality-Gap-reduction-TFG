import torch
import torch.nn as nn
import os

from .ddpg import DDPGAgent
from .configuration import Configuration

# Instantiate DDPGAgent and Configuration
ddpg_model = DDPGAgent()
config = Configuration()

model_path = os.path.expanduser('~/tfg/Simulation/src/models/')
model_name = config.model_name

train_or_pretrained = input("Do you want to 'train' or use a 'pretrained' model?")

if train_or_pretrained == "pretrained":
    print(f"using pretrained model located in {model_path} ...")

    ddpg_model.load_state_dict(torch.load(model_path + model_name))
    ddpg_model.eval()

    for param in ddpg_model.layer1.parameters():
        param.requires_grad = False

    for param in ddpg_model.layer2.parameters():
        param.requires_grad = False

elif train_or_pretrained == "train":
    print(f"training ddpg model from scratch...")

# Print model parameters and layers
print("\nModel Parameters:")
for name, param in ddpg_model.named_parameters():
    print(f"{name}: {param.shape}")

print("\nModel Layers:")
for name, layer in ddpg_model.named_children():
    print(f"{name}: {layer}")
