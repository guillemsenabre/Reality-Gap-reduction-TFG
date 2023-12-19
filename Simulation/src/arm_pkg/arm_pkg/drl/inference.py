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

    train_or_pretrained = input("Hey do you want to 'train' from scratch or use a 'pretrained' model?")

    if train_or_pretrained == "pretrained":
        get_pretrained_model()

    elif train_or_pretrained == "train":
        print(f"training ddpg model from scratch...")
        get_plain_model()

    else:
        print("STFU goodbye")


def get_pretrained_model():
    print(f"using pretrained model located in {model_path} ...")

    model_name = config.model_name
    model_path = os.path.expanduser(f'~/tfg/Simulation/src/models/{model_name}')
        
    # Load the state dictionaries
    checkpoint = torch.load('ddpg_model.pth')

    # Load the state dictionaries into the actor and critic models
    ddpg_model.actor.load_state_dict(checkpoint['actor_state_dict'])
    ddpg_model.critic.load_state_dict(checkpoint['critic_state_dict'])

def get_plain_model():




if __name__=='__main__':
    main()