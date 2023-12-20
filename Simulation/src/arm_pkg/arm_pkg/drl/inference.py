import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import os
import time

from ddpg import DDPGAgent
from configuration import Configuration
from reset import Reset
from sub_modules.move_joints import MoveJoints

class Inference(Node):
    def __init__(self):
        super().__init__('inference')

        self.get_logger().info("Starting inference node...")

        time.sleep(100)

        self.get_logger().info("Getting esp32 ports...")

        self.port1 = self.config.port1
        self.port2 = self.config.port2

        self.move = MoveJoints()
        self.config = Configuration()
        self.reset = Reset()
        self.ddpg_model = DDPGAgent(self.config.state_dim, self.config.action_dim)
                
        train_or_pretrained = input("Hey do you want to 'train' from scratch or use a 'pretrained' model?")

        if train_or_pretrained == "pretrained":
            self.get_logger().info("Getting pretrained model ready...")
            self.get_pretrained_model()
            #self.train()??

        elif train_or_pretrained == "train":
            self.get_logger().info((f"Training ddpg model from scratch..."))
            self.train(self.ddpg_model)

        else:
            self.get_logger().info(("STFU goodbye"))


    def get_pretrained_model(self):
        self.get_logger().info((f"Using pretrained model located in {model_path} ..."))

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
        #TODO - state = 

        # - 10 servo motor angles
        # - 2 HSCR04 distances
        # - 3 quaternions from 3 IMUs

        #action = self.ddpg_model.select_action(state)
        #self.move(action)
        #TODO - next_state = 
        #TODO - reward = 
        #TODO - terminal_condition

        # - Add safety protocols
        # - Velocity, object drop, base join angle,...
        # - Add a reset joints, to position the joints at 0.

        #self.ddpg_model.update(state, action, reward, next_state, terminal_condition)
        pass





if __name__ == '__main__':
    rclpy.init()
    inference = Inference()
    rclpy.spin(inference)
    inference.destroy_node()
    rclpy.shutdown()