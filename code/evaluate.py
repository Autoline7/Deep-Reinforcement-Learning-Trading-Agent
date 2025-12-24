import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.data_preprocessing import get_train_test_data
from env.trading_env import TradingEnv
from agent.dqn import DQN
import random
import json
import os


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_agent(env, model, device):
    state, _ = env.reset()
    done = False
    portfolio_values = []
    actions = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

            if env.position == 0:
                q_values[0, 2] = -float('inf')  # Block Sell
            else:
                q_values[0, 1] = -float('inf')  # Block Buy

            action = q_values.argmax(dim=1).item()

        next_state, reward, done, truncated, info = env.step(action)

        portfolio_values.append(env._get_portfolio_value())
        actions.append(action)
        state = next_state

    return portfolio_values, actions


def calculate_returns(values):
    """Calculate daily returns from portfolio values."""
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    return returns


def calculate_risk_metrics(agent_returns, benchmark_returns):
    """
    Calculate Beta
    """
    # Calculate beta using covariance/variance
    cov_matrix = np.cov(agent_returns, benchmark_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return beta


def get_optimal_actions(prices):
    """
    Generate 'optimal' actions (for confusion matrix).
    """
    optimal_actions = []
    position = 0  # 0 = no position, 1 = holding

    for i in range(len(prices) - 1):
        future_return = (prices[i + 1] - prices[i]) / prices[i]

        if position == 0:
            if future_return > 0.001:  # Threshold for meaningful gain
                optimal_actions.append(1)  # Buy
                position = 1
            else:
                optimal_actions.append(0)  # Hold
        else:
            if future_return < -0.001:  # Threshold for meaningful loss
                optimal_actions.append(2)  # Sell
                position = 0
            else:
                optimal_actions.append(0)  # Hold

    if position == 1:
        optimal_actions.append(2)
    else:
        optimal_actions.append(0)

    return optimal_actions


def plot_confusion_matrix(agent_actions, optimal_actions, num_episodes):
    """
    Plot confusion matrix comparing agent actions to optimal actions.
    """
    min_len = min(len(agent_actions), len(optimal_actions))
    agent_actions = agent_actions[:min_len]
    optimal_actions = optimal_actions[:min_len]

    labels = ['Hold', 'Buy', 'Sell']

    cm = confusion_matrix(optimal_actions, agent_actions, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    plt.title(f"Action Confusion Matrix (Agent vs Optimal)\nTrained on {num_episodes} episodes")
    plt.ylabel("Optimal Action (Hindsight)")
    plt.xlabel("Agent Action")
    plt.tight_layout()
    plt.show()

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0

    print(f"\n===== CONFUSION MATRIX ANALYSIS =====")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    for i, label in enumerate(labels):
        if cm[:, i].sum() > 0:
            precision = cm[i, i] / cm[:, i].sum()
        else:
            precision = 0
        if cm[i, :].sum() > 0:
            recall = cm[i, i] / cm[i, :].sum()
        else:
            recall = 0
        print(f"{label}: Precision={precision:.2f}, Recall={recall:.2f}")


def get_num_episodes(model_path):
    """
    Try to get number of training episodes from various sources.
    """
    results_dir = os.path.dirname(model_path)
    possible_files = [
        os.path.join(results_dir, 'training_config.json'),
        os.path.join(results_dir, 'training_log.json'),
        os.path.join(results_dir, 'config.json'),
        os.path.join(results_dir, 'hyperparams.json'),
    ]

    for filepath in possible_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                    if 'num_episodes' in config:
                        return config['num_episodes']
                    if 'episodes' in config:
                        return config['episodes']
                    if 'n_episodes' in config:
                        return config['n_episodes']
            except:
                pass

    model_name = os.path.basename(model_path)
    import re
    match = re.search(r'(\d+)(?:ep|episodes?|_)', model_name)
    if match:
        return int(match.group(1))

    return "Unknown"


def evaluate_dqn(model_path="results/dqn_spy.pth", ticker="SPY", seed=42, num_episodes=None):
    """
    Enhanced evaluation with confusion matrix and active return.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if num_episodes is None:
        num_episodes = get_num_episodes(model_path)

    print(f"Model trained on: {num_episodes} episodes")

    _, test_df = get_train_test_data(ticker)

    env = TradingEnv(test_df)
    env.transaction_cost = 0.00

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = DQN(input_dim=obs_dim, output_dim=n_actions)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Running agent on test data...")
    portfolio_values, actions = run_agent(env, model, device)

    # --- TRADE LOG ---
    print("\n===== TRADE LOG =====")
    print(f"{'Step':<6} | {'Date':<12} | {'Action':<6} | {'Price':<10}")
    print("-" * 45)

    trade_count = 0
    for i, action in enumerate(actions):
        if action == 1 or action == 2:
            trade_count += 1
            date_str = test_df.iloc[i]["Date"].strftime("%Y-%m-%d")
            price = test_df.iloc[i]["Close"]
            action_str = "BUY" if action == 1 else "SELL"
            print(f"{i:<6} | {date_str:<12} | {action_str:<6} | ${price:.2f}")

    print("-" * 45)
    print(f"Total Trades: {trade_count}")

    # --- BASIC METRICS ---
    initial_price = test_df.iloc[0]["Close"]
    buy_hold_values = [10000 * (price / initial_price) for price in test_df["Close"]]

    agent_return = (portfolio_values[-1] - 10000) / 10000
    bnh_return = (buy_hold_values[-1] - 10000) / 10000

    # --- Alpha ---
    active_return = agent_return - bnh_return

    print("\n===== PERFORMANCE METRICS =====")
    print(f"Agent Return:      {agent_return * 100:.2f}%")
    print(f"Buy & Hold Return: {bnh_return * 100:.2f}%")
    print(f"Active Return:     {active_return * 100:.2f}%")
    print(f"Final Value:       ${portfolio_values[-1]:.2f}")

    # --- CALCULATE RETURNS ---
    min_len = min(len(portfolio_values), len(buy_hold_values))
    portfolio_values_aligned = portfolio_values[:min_len]
    buy_hold_values_aligned = buy_hold_values[:min_len]

    agent_returns = calculate_returns(portfolio_values_aligned)
    benchmark_returns = calculate_returns(buy_hold_values_aligned)

    print(f"\nData points: {min_len} days, {len(agent_returns)} return observations")

    # --- BETA ---
    beta = calculate_risk_metrics(agent_returns, benchmark_returns)

    print("\n===== RISK ANALYSIS =====")
    print(f"Beta: {beta:.3f}")

    if beta > 1:
        print(f"→ Beta > 1 means the agent is more volatile than the market")
    elif beta < 1:
        print(f"→ Beta < 1 means the agent is less volatile than the market")

    # --- SHARPE RATIO (OVERALL) ---
    if np.std(agent_returns) > 0:
        overall_sharpe_agent = (np.mean(agent_returns) / np.std(agent_returns)) * np.sqrt(252)
    else:
        overall_sharpe_agent = 0

    if np.std(benchmark_returns) > 0:
        overall_sharpe_benchmark = (np.mean(benchmark_returns) / np.std(benchmark_returns)) * np.sqrt(252)
    else:
        overall_sharpe_benchmark = 0

    print(f"\n===== SHARPE RATIO (OVERALL) =====")
    print(f"Agent Sharpe Ratio (annualized): {overall_sharpe_agent:.3f}")
    print(f"Benchmark Sharpe Ratio (annualized): {overall_sharpe_benchmark:.3f}")

    # --- PLOTS ---
    # 1. Portfolio Performance Only
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values_aligned, label="DQN Agent", color='blue')
    plt.plot(buy_hold_values_aligned, label="Buy & Hold", color='gray', linestyle='--')
    plt.title(f"Agent vs Market (Test Set)\nTrained on {num_episodes} episodes")
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid()
    plt.show()

    # 2. Confusion Matrix
    prices = test_df["Close"].values[:min_len]
    optimal_actions = get_optimal_actions(prices)
    actions_aligned = actions[:min_len]
    plot_confusion_matrix(actions_aligned, optimal_actions, num_episodes)

    return portfolio_values_aligned, actions_aligned


if __name__ == "__main__":
    evaluate_dqn(num_episodes=500)