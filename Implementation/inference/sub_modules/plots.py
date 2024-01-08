import matplotlib.pyplot as plt
import numpy as np

def plot_results(episode_rewards, actor_losses, critic_losses):
    # Flatten the list of lists into a single list
    rewards = [reward for reward in episode_rewards]

    # Quadratic interpolation for rewards
    steps_rewards = np.arange(len(rewards))
    quadratic_interp_rewards = np.polyfit(steps_rewards, rewards, 2)
    quadratic_curve_rewards = np.poly1d(quadratic_interp_rewards)

    # Quadratic interpolation for actor losses
    steps_actor_losses = np.arange(len(actor_losses))
    quadratic_interp_actor_losses = np.polyfit(steps_actor_losses, actor_losses, 2)
    quadratic_curve_actor_losses = np.poly1d(quadratic_interp_actor_losses)

    # Quadratic interpolation for critic losses
    steps_critic_losses = np.arange(len(critic_losses))
    quadratic_interp_critic_losses = np.polyfit(steps_critic_losses, critic_losses, 2)
    quadratic_curve_critic_losses = np.poly1d(quadratic_interp_critic_losses)

    # Plot episode rewards, actor losses, and critic losses
    plt.figure(figsize=(12, 6))

    plt.plot(steps_rewards, quadratic_curve_rewards(steps_rewards), label='Smoothed Episode Total Reward', linestyle='-', marker='o', markersize=3)
    plt.title('Smoothed Episode Rewards')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    plt.plot(steps_actor_losses, quadratic_curve_actor_losses(steps_actor_losses), label='Smoothed Actor Loss', linestyle='-', alpha=0.7)
    plt.plot(steps_critic_losses, quadratic_curve_critic_losses(steps_critic_losses), label='Smoothed Critic Loss', linestyle='--', alpha=0.7)
    plt.title('Smoothed Actor and Critic Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
