import torch
import torch.nn as nn

from sub_modules.ddpg import DDPGAgent
from sub_modules.abort_save import AbortOrSave
from sub_modules.move_joints import MoveJoints
from sub_modules.states import States
from sub_modules.reward import Reward
from sub_modules.plots import plot_results

class Main:
    def __init__(self):
        print("Initializing variables...")

        self.model_name = "ddpg_model.pth"
        self.number_motors = 1
        self.number_sensors = 3
        self.episodes = 10
        self.episode_rewards = []
        self.port = input("Select Serial port: ")

        print(f"Port {self.port} selected")
        print("Initializing modules...")

        self.move = MoveJoints()
        self.states = States()
        self.reward = Reward()
        self.abort = AbortOrSave()

        print("select network dimensions:")
        state_dim = int(input("Select state dim --> "))
        action_dim = int(input("Select action dimensions --> "))
        self.ddpg_model = DDPGAgent(state_dim, action_dim)
                
        train_or_pretrained = input("Hey, do you want to 'train' from scratch or use a 'pretrained' model? ")
        print(f"Decision made! {train_or_pretrained}")

        if train_or_pretrained == "pretrained":
            print("Getting pretrained model ready...")
            self.get_pretrained_model()

        elif train_or_pretrained == "train":
            print("Training ddpg model from scratch...")
            
            self.train()

        else:
            print("STFU goodbye")

    def get_pretrained_model(self):
        model_name = self.model_name
        model_path = f"../models/{model_name}"    # RELATIVE

        # Load the state dictionaries
        print(model_path)
        pretrained = torch.load(model_path, weights_only = True)
        print(pretrained)

        # Load the state dictionaries into the actor and critic models
        self.ddpg_model.actor.load_state_dict(pretrained['actor_state_dict'])
        self.ddpg_model.critic.load_state_dict(pretrained['critic_state_dict'])

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
    
    def next_episode(self):
        # Giving the option to reset the motors (to 90º) or random (#TODO -)
        print("Do you want to keep training or see the results?")
        begin = input("Type 't' or 'r'")

        if begin == 'train':
            self.episode_rewards.append([0.0, 0.0, 0.0]) # So next episode rewards are visible
            self.move.reset_motors()

        elif begin == 'results':
            plot_results(self.episode_rewards, self.ddpg_model.actor_losses, self.ddpg_model.critic_losses)

        else:
            self.episodes -= 1

    def train(self):
        while self.episodes != 0:
            while True:
                print("Getting states...")
                states = self.states.read_sensor_data(self.port, self.number_motors, self.number_sensors)
                print(states)

                # - 10 servo motor angles
                # - 2 HSCR04 distances
                # - Euler orientations from 1 IMUs (can be expanded to handle quaternions and more IMUs)

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

                print("Updating model...")
                self.ddpg_model.update(states, action, reward, next_state, terminal_condition)

                # Updating rewards
                self.episode_rewards.append(reward)
                
                if terminal_condition:
                    break

            self.next_episode()

            

if __name__ == '__main__':
    Main()
