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

    print(f"Random seed set to: {seed}")


def run_agent(env, model, device):
    state, _ = env.reset()
    done = False
    portfolio_values = []
    actions = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

            # --- MASKING LOGIC START ---
            if env.position == 0:
                q_values[0, 2] = -float('inf')  # Block Sell
            else:
                q_values[0, 1] = -float('inf')  # Block Buy
            # --- MASKING LOGIC END ---

            action = q_values.argmax(dim=1).item()

        next_state, reward, done, truncated, info = env.step(action)

        portfolio_values.append(env._get_portfolio_value())
        actions.append(action)
        state = next_state

    return portfolio_values, actions


def max_drawdown(values):
    """
    Computes maximum drawdown of a portfolio value series.
    """
    values = np.array(values)
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks
    return drawdowns.min()


def calculate_returns(values):
    """Calculate daily returns from portfolio values."""
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    return returns


def calculate_rolling_sharpe(returns, window=20, risk_free_rate=0.0):
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of daily returns
        window: Rolling window size (default 20 trading days ~ 1 month)
        risk_free_rate: Daily risk-free rate (default 0)

    Returns:
        Array of rolling Sharpe ratios (annualized)
    """
    rolling_sharpe = []

    for i in range(len(returns)):
        if i < window - 1:
            rolling_sharpe.append(np.nan)
        else:
            window_returns = returns[i - window + 1:i + 1]
            excess_returns = window_returns - risk_free_rate

            if np.std(window_returns) > 0:
                # Annualize: multiply by sqrt(252) for daily returns
                sharpe = (np.mean(excess_returns) / np.std(window_returns)) * np.sqrt(252)
            else:
                sharpe = 0.0
            rolling_sharpe.append(sharpe)

    return np.array(rolling_sharpe)


def calculate_alpha(agent_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Calculate Jensen's Alpha using CAPM.

    Alpha = R_p - [R_f + beta * (R_m - R_f)]

    Args:
        agent_returns: Array of agent's daily returns
        benchmark_returns: Array of benchmark's daily returns
        risk_free_rate: Daily risk-free rate

    Returns:
        alpha (annualized), beta, r_squared
    """
    # Calculate beta using covariance/variance
    cov_matrix = np.cov(agent_returns, benchmark_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Calculate expected return using CAPM
    avg_agent_return = np.mean(agent_returns)
    avg_benchmark_return = np.mean(benchmark_returns)

    expected_return = risk_free_rate + beta * (avg_benchmark_return - risk_free_rate)

    # Alpha (daily)
    alpha_daily = avg_agent_return - expected_return

    # Annualize alpha (252 trading days)
    alpha_annualized = alpha_daily * 252

    # R-squared
    correlation = np.corrcoef(agent_returns, benchmark_returns)[0, 1]
    r_squared = correlation ** 2

    return alpha_annualized, beta, r_squared


def get_optimal_actions(prices):
    """
    Generate 'optimal' actions based on hindsight (for confusion matrix).
    Buy before price goes up, Sell before price goes down, Hold otherwise.

    Returns:
        List of optimal actions: 0=Hold, 1=Buy, 2=Sell
    """
    optimal_actions = []
    position = 0  # 0 = no position, 1 = holding

    for i in range(len(prices) - 1):
        future_return = (prices[i + 1] - prices[i]) / prices[i]

        if position == 0:
            # Not holding - should we buy?
            if future_return > 0.001:  # Threshold for meaningful gain
                optimal_actions.append(1)  # Buy
                position = 1
            else:
                optimal_actions.append(0)  # Hold (stay out)
        else:
            # Holding - should we sell?
            if future_return < -0.001:  # Threshold for meaningful loss
                optimal_actions.append(2)  # Sell
                position = 0
            else:
                optimal_actions.append(0)  # Hold (stay in)

    # Last action is always hold or sell based on position
    if position == 1:
        optimal_actions.append(2)  # Sell at end
    else:
        optimal_actions.append(0)  # Hold

    return optimal_actions


def plot_confusion_matrix(agent_actions, optimal_actions, num_episodes):
    """
    Plot confusion matrix comparing agent actions to optimal actions.
    """
    # Ensure same length
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

    # Print accuracy metrics
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


def plot_drawdown(portfolio_values, num_episodes):
    values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, color='red', alpha=0.7)
    plt.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
    plt.title(f"Portfolio Drawdown (Risk Analysis)\nTrained on {num_episodes} episodes")
    plt.ylabel("Drawdown %")
    plt.xlabel("Days")
    plt.grid()
    plt.show()

    print(f"Max Drawdown: {drawdown.min() * 100:.2f}%")


def plot_rolling_sharpe(agent_returns, benchmark_returns, num_episodes, window=20):
    """Plot rolling Sharpe ratio for agent vs benchmark."""
    agent_rolling_sharpe = calculate_rolling_sharpe(agent_returns, window)
    benchmark_rolling_sharpe = calculate_rolling_sharpe(benchmark_returns, window)

    plt.figure(figsize=(12, 5))
    plt.plot(agent_rolling_sharpe, label='DQN Agent', color='blue', alpha=0.8)
    plt.plot(benchmark_rolling_sharpe, label='Buy & Hold', color='gray', linestyle='--', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1')
    plt.axhline(y=-1, color='red', linestyle=':', alpha=0.5, label='Sharpe = -1')

    plt.title(f"Rolling Sharpe Ratio ({window}-day window, Annualized)\nTrained on {num_episodes} episodes")
    plt.xlabel("Days")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary stats
    valid_agent = agent_rolling_sharpe[~np.isnan(agent_rolling_sharpe)]
    valid_benchmark = benchmark_rolling_sharpe[~np.isnan(benchmark_rolling_sharpe)]

    print(f"\n===== ROLLING SHARPE STATISTICS =====")
    print(f"Agent - Mean: {np.mean(valid_agent):.2f}, Std: {np.std(valid_agent):.2f}")
    print(f"Benchmark - Mean: {np.mean(valid_benchmark):.2f}, Std: {np.std(valid_benchmark):.2f}")
    print(f"% of days Agent Sharpe > Benchmark: {(valid_agent > valid_benchmark).mean() * 100:.1f}%")


def get_num_episodes(model_path):
    """
    Try to get number of training episodes from various sources.
    """
    # Try to find a training log or config file
    results_dir = os.path.dirname(model_path)

    # Check for common log file patterns
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

    # If no config found, try to extract from model filename
    # e.g., "dqn_spy_500ep.pth" or "model_1000.pth"
    model_name = os.path.basename(model_path)
    import re
    match = re.search(r'(\d+)(?:ep|episodes?|_)', model_name)
    if match:
        return int(match.group(1))

    # Default value if nothing found
    return "Unknown"


def evaluate_dqn(model_path="results/dqn_spy.pth", ticker="SPY", seed=42, num_episodes=None):
    """
    Enhanced evaluation with confusion matrix, alpha, and rolling Sharpe ratio.

    Args:
        model_path: Path to trained model
        ticker: Stock ticker symbol
        seed: Random seed
        num_episodes: Number of training episodes (will try to auto-detect if None)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get number of episodes
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

    print("\n===== PERFORMANCE METRICS =====")
    print(f"Agent Return:      {agent_return * 100:.2f}%")
    print(f"Buy & Hold Return: {bnh_return * 100:.2f}%")
    print(f"Final Value:       ${portfolio_values[-1]:.2f}")

    # --- CALCULATE RETURNS ---
    # Ensure both arrays have the same length
    min_len = min(len(portfolio_values), len(buy_hold_values))
    portfolio_values_aligned = portfolio_values[:min_len]
    buy_hold_values_aligned = buy_hold_values[:min_len]

    agent_returns = calculate_returns(portfolio_values_aligned)
    benchmark_returns = calculate_returns(buy_hold_values_aligned)

    print(f"\nData points: {min_len} days, {len(agent_returns)} return observations")

    # --- ALPHA & BETA ---
    alpha, beta, r_squared = calculate_alpha(agent_returns, benchmark_returns)

    print("\n===== ALPHA & BETA ANALYSIS =====")
    print(f"Jensen's Alpha (annualized): {alpha * 100:.2f}%")
    print(f"Beta: {beta:.3f}")
    print(f"R-squared: {r_squared:.3f}")

    if alpha > 0:
        print("→ Positive alpha indicates the agent outperformed on a risk-adjusted basis")
    else:
        print("→ Negative alpha indicates the agent underperformed on a risk-adjusted basis")

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

    # 1. Portfolio Performance
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values_aligned, label="DQN Agent", color='blue')
    plt.plot(buy_hold_values_aligned, label="Buy & Hold", color='gray', linestyle='--')
    plt.title(f"Agent vs Market (Test Set)\nTrained on {num_episodes} episodes")
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid()
    plt.show()

    # 2. Drawdown
    plot_drawdown(portfolio_values_aligned, num_episodes)

    # 3. Rolling Sharpe Ratio
    plot_rolling_sharpe(agent_returns, benchmark_returns, num_episodes, window=20)

    # 4. Confusion Matrix
    prices = test_df["Close"].values[:min_len]
    optimal_actions = get_optimal_actions(prices)
    actions_aligned = actions[:min_len]
    plot_confusion_matrix(actions_aligned, optimal_actions, num_episodes)

    return portfolio_values_aligned, actions_aligned


if __name__ == "__main__":
    evaluate_dqn(num_episodes=500)