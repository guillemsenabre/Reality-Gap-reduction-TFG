import torch
import torch.nn as nn
import os
import time

from ddpg import DDPGAgent
from configuration import Configuration
from sub_modules.abort_save import AbortOrSave
from sub_modules.move_joints import MoveJoints
from sub_modules.states import States
from sub_modules.reward import Reward

class Inference:
    def __init__(self):
        print("Starting inference node...")

        self.config = Configuration()
        self.move = MoveJoints()
        self.states = States()
        self.reward = Reward()
        self.abort = AbortOrSave()
        self.save = AbortOrSave()
        self.config.dimensions()
        self.ddpg_model = DDPGAgent(self.config.state_dim, self.config.action_dim)
                
        train_or_pretrained = input("Hey, do you want to 'train' from scratch or use a 'pretrained' model? ")

        if train_or_pretrained == "pretrained":
            print("Getting pretrained model ready...")
            self.get_pretrained_model()
            #self.train()??

        elif train_or_pretrained == "train":
            print("Training ddpg model from scratch...")
            self.train(self.ddpg_model)

        else:
            print("STFU goodbye")

    def get_pretrained_model(self):
        model_name = self.config.model_name
        model_path = os.path.expanduser(f'~/tfg/Simulation/src/models/{model_name}')

        # Load the state dictionaries
        checkpoint = torch.load('ddpg_model.pth')

        # Load the state dictionaries into the actor and critic models
        self.ddpg_model.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.ddpg_model.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Freeze the first two layers (fc1 and fc2) of the actor model
        for param in self.ddpg_model.actor.fc1.parameters():
            param.requires_grad = False
        for param in self.ddpg_model.actor.fc2.parameters():
            param.requires_grad = False

        # Freeze the first two layers (fc1 and fc2) of the critic model
        for param in self.ddpg_model.critic.fc1.parameters():
            param.requires_grad = False
        for param in self.ddpg_model.critic.fc2.parameters():
            param.requires_grad = False

    def train(self, model):
        state = self.states.read_sensor_data

        # - 10 servo motor angles
        # - 2 HSCR04 distances
        # - 3 quaternions from 3 IMUs

        prev_angles = state[:10] #dynamic velocity reward

        action = self.ddpg_model.select_action(state)
        self.move(action)
        next_state = self.states.read_sensor_data()
        reward = self.reward(prev_angles)
        terminal_condition = self.abort.terminal_condition()

        # - Add safety protocols
        # - Velocity, object drop, base join angle,...
        # - Add a reset joints, to position the joints at 0.

        self.ddpg_model.update(state, action, reward, next_state, terminal_condition)


if __name__ == '__main__':
    Inference()
