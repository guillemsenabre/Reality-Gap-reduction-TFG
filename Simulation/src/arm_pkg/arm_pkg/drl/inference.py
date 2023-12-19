import torch
import torch.nn as nn
import os

from ddpg import DDPGAgent
from configuration import Configuration


# Instantiate DDPGAgent and Configuration
config = Configuration()
ddpg_model = DDPGAgent(config.state_dim, config.action_dim)


def main():
    print("FUCKING STARTED")

    train_or_pretrained = input("Do you want to 'train' or use a 'pretrained' model?")

    if train_or_pretrained == "pretrained":
        get_pretrained_model()

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


def get_pretrained_model():
    print(f"using pretrained model located in {model_path} ...")

    model_name = config.model_name
    model_path = os.path.expanduser(f'~/tfg/Simulation/src/models/{model_name}')
        
    # Load the state dictionaries
    checkpoint = torch.load('ddpg_model.pth')

    # Load the state dictionaries into the actor and critic models
    ddpg_model.actor.load_state_dict(checkpoint['actor_state_dict'])
    ddpg_model.critic.load_state_dict(checkpoint['critic_state_dict'])



if __name__=='__main__':
    main()