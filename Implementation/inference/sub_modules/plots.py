import matplotlib.pyplot as plt

def plot_results(episode_rewards, actor_losses, critic_losses):
    # Flatten the list of lists into a single list
    rewards = [reward for reward in episode_rewards]

    # Plot episode rewards
    plt.figure(figsize=(12, 6))

    plt.plot(rewards, label='Episode Total Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.legend()

    # Plot actor and critic losses
    plt.plot(actor_losses, label='Actor Loss', alpha=0.7)
    plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
    plt.title('Actor and Critic Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()