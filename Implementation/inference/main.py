import torch
import torch.nn as nn
import os
import time

from sub_modules.ddpg import DDPGAgent
from sub_modules.abort_save import AbortOrSave
from sub_modules.move_joints import MoveJoints
from sub_modules.states import States
from sub_modules.reward import Reward

class Main:
    def __init__(self):
        print("Initializing variables...")

        self.model_name = "ddpg_model.pth"
        self.number_motors = 1
        self.number_sensors = 3
        self.port = "/dev/ttyUSB0"

        print("Initializing modules...")

        self.move = MoveJoints()
        self.states = States()
        self.reward = Reward()
        self.abort = AbortOrSave()
        time.sleep(0.3)

        print("selecting network dimensions...")
        state_dim = int(input("Select state dim --> "))
        action_dim = int(input("Select action dimensions --> "))
        self.ddpg_model = DDPGAgent(state_dim, action_dim)
                
        train_or_pretrained = input("Hey, do you want to 'train' from scratch or use a 'pretrained' model? ")

        if train_or_pretrained == "pretrained":
            print("Getting pretrained model ready...")
            self.get_pretrained_model()

            begin = input("Type 'start' to begin training and initialize the motors")
            if begin == 'start':
                
            self.train()

        elif train_or_pretrained == "train":
            print("Training ddpg model from scratch...")
            
            self.train()

        else:
            print("STFU goodbye")

    def get_pretrained_model(self):
        model_name = self.model_name
        model_path = os.path.expanduser(f'~/tfg/Simulation/src/models/{model_name}')

        # Load the state dictionaries
        checkpoint = torch.load(model_path)

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

    def train(self):
        while True:
            print("Getting states...")
            states = self.states.read_sensor_data(self.port, self.number_motors, self.number_sensors)
            print(states)

            # - 10 servo motor angles
            # - 2 HSCR04 distances
            # - 3 quaternions from 3 IMUs (for now its 3 euler angles)

            print("Getting angles...")
            prev_angles = states[:10] #dynamic velocity reward
            print(prev_angles)
            print("Passing states to ddpg...")
            action = self.ddpg_model.select_action(states)
            print(action)
            self.move.move_joints(action, self.port, self.number_motors)
            print("Getting new states...")
            next_state = self.states.read_sensor_data()
            print("Getting new angles...")
            current_angles = states[:self.number_motors] #for the terminal condition
            print("Calculating reward...")
            reward = self.reward.reward(prev_angles, states, self.number_motors)
            print("Getting terminal condition status...")
            terminal_condition = self.abort.terminal_condition(current_angles, reward)

            # - Add safety protocols
            # - Velocity, object drop, base join angle,...
            # - Add a reset joints, to position the joints at 0.

            print("Updating model...")

            self.ddpg_model.update(states, action, reward, next_state, terminal_condition)

            if terminal_condition:
                break


if __name__ == '__main__':
    Main()
