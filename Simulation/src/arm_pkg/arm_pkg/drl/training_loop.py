import rclpy

from .rosdata import RosData
from .reset import Reset
from .configuration import Configuration
from .plots import plot_results

import numpy as np
import torch


#TODO - Tune hyperparameters (lr)
#TODO - Understanding losses behaviour
#TODO - Save model 



#SECTION - TRAINING LOOP -



def main(args=None):
    rclpy.init(args=args)
    config = Configuration()
    ros_data = RosData()
    reset = Reset()
    num_episodes = config.num_episodes

    episode_rewards = []

    for episode in range(num_episodes):

        print(f'Running poch: {episode}')

        reset.reset()
        
        # Waiting for the first state message to be received
        while not ros_data.state.any():
            print("Waiting for state data ...")
            rclpy.spin_once(ros_data)

        print("Training!")

        episode_reward_list = []

        while True:
            state = ros_data.state
            action = ros_data.agent.select_action(state)
            ros_data.move_joints(action)
            next_state = ros_data.state
            reward = ros_data.reward_value

            terminal_condition = ros_data.terminal_condition()

            # Update agent
            ros_data.agent.update(state, action, reward, next_state, terminal_condition)

            # Collect data for the current episode
            episode_reward_list.append(reward)

            rclpy.spin_once(ros_data)

            if terminal_condition:
                print(f'Terminal condition reached!')
                break
            
            elif len(episode_reward_list) == 50:
                avg_reward = np.mean(episode_reward_list)
                if avg_reward <= 0.6:
                    torch.save(ros_data.agent, 'ddpg_model.pth')
                    print(f'Model saved due to average reward less than 0.6: {avg_reward}')


        # Store episode data for plotting
        episode_rewards.append(episode_reward_list)

        # Plot at the end of each episode
        plot_results(episode_rewards, ros_data.agent.actor_losses, ros_data.agent.critic_losses)


    ros_data.destroy_node()
    rclpy.shutdown()

#!SECTION
#!SECTION

if __name__ == '__main__':
    main() 