import random
import numpy as np
import torch

from utils.data_preprocessing import get_train_test_data
from env.trading_env import TradingEnv
from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior on GPU (if you use one)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: Force deterministic algorithms (can be slower)
    # torch.use_deterministic_algorithms(True)

    print(f"Random seed set to: {seed}")

def select_action(state, policy_net, env, epsilon, device):
    """
    Selects action with Masking:
    - If Position = 0 (No shares): Can only HOLD (0) or BUY (1). SELL (2) is blocked.
    - If Position = 1 (Has shares): Can only HOLD (0) or SELL (2). BUY (1) is blocked.
    """
    # 1. Determine which actions are allowed right now
    valid_actions = [0]  # Hold is always allowed
    if env.position == 0:
        valid_actions.append(1)  # Can Buy
    else:
        valid_actions.append(2)  # Can Sell

    # 2. Epsilon-Greedy Logic
    if random.random() < epsilon:
        # EXPLORE: Randomly pick from VALID actions only
        return random.choice(valid_actions)
    else:
        # EXPLOIT: Ask the Neural Network
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)

            # MASKING: Set Q-values of invalid actions to negative infinity
            # This ensures argmax() NEVER picks them.
            if env.position == 0:
                q_values[0, 2] = -float('inf')  # Block Sell
            else:
                q_values[0, 1] = -float('inf')  # Block Buy

            return q_values.argmax(dim=1).item()

def train_dqn(
    ticker="SPY",
    num_episodes=500,
    gamma=0.99,
    batch_size=64,
    buffer_capacity=100_000,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.01,  # LOWERED from 0.05
    epsilon_decay=0.99,  # FASTER DECAY (was 0.995)
    target_update_freq=1000,
    seed=42
    ):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------- Load data and create environment ---------
    train_df, test_df = get_train_test_data(ticker)

    env = TradingEnv(train_df)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --------- Initialize networks ---------
    policy_net = DQN(input_dim=obs_dim, output_dim=n_actions, lr=lr).to(device)
    target_net = DQN(input_dim=obs_dim, output_dim=n_actions, lr=lr).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # target net is not trained directly

    optimizer = policy_net.optimizer
    loss_fn = policy_net.loss_fn

    # --------- Replay buffer ---------
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # --------- Training loop ---------
    epsilon = epsilon_start
    total_steps = 0

    episode_rewards = []
    episode_final_values = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 1. Choose action
            action = select_action(state, policy_net, env, epsilon, device)

            # 2. Step environment
            next_state, reward, done, truncated, info = env.step(action)

            # 3. Store transition
            replay_buffer.push(state, action, float(reward), next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # 4. Sample from buffer and train
            if len(replay_buffer) >= batch_size:
                (
                    states_np,
                    actions_np,
                    rewards_np,
                    next_states_np,
                    dones_np,
                ) = replay_buffer.sample(batch_size)

                states = torch.tensor(states_np, dtype=torch.float32, device=device)
                actions = torch.tensor(actions_np, dtype=torch.int64, device=device)
                rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
                dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

                # Current Q-values: Q(s, a) from policy_net
                q_values = policy_net(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target Q-values: r + gamma * max_a' Q_target(s', a')
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1)[0]
                    q_targets = rewards + gamma * next_q_values * (1.0 - dones)

                loss = loss_fn(q_values, q_targets)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

                optimizer.step()

            # 5. Update target network periodically
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # ----- End of episode -----
        episode_rewards.append(episode_reward)
        episode_final_values.append(env._get_portfolio_value())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(
            f"Episode {episode}/{num_episodes} | "
            f"Reward: {episode_reward:.4f} | "
            f"Final Value: {env._get_portfolio_value():.2f} | "
            f"Epsilon: {epsilon:.4f}"
        )


    torch.save(policy_net.state_dict(), "results/dqn_spy.pth")
    np.save("results/episode_rewards.npy", np.array(episode_rewards))
    np.save("results/episode_final_values.npy", np.array(episode_final_values))

    print("Training finished. Model and logs saved to 'results/'.")


if __name__ == "__main__":
    train_dqn(num_episodes=500)
