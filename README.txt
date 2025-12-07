================================================================================
Deep Reinforcement Learning Trading Agent (DQN)
================================================================================

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements a Deep Q-Network (DQN) agent designed to autonomously
trade the S&P 500 (SPY). The agent interacts with a custom OpenAI Gym
environment to learn optimal execution strategies (Buy, Sell, Hold) based on
market data and technical analysis.

The system utilizes PyTorch for the neural network, utilizing Experience Replay
and Target Networks to stabilize training. It features a custom "Action Masking"
logic to ensure the agent never attempts invalid trades (e.g., selling shares
it does not own).

2. KEY FEATURES
--------------------------------------------------------------------------------
- **Deep Q-Network (DQN):** 2-layer neural network with ReLU activation.
- **Custom Environment:** Simulates a brokerage account with transaction costs.
- **Feature Engineering:** Inputs include normalized RSI, MACD, Bollinger Bands,
  SMA, VIX, TNX (Interest Rates), ATR (Volatility), and OBV (Momentum).
- **Risk Management:** Action masking prevents illegal trades; reward function
  penalizes holding during downturns.
- **Evaluation:** Calculates Rolling Sharpe Ratio, Maximum Drawdown, and
  Jensen's Alpha against a Buy & Hold benchmark.

3. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
To run this project, ensure your files are organized into the following folders,
as the Python scripts use relative imports (e.g., `from agent.dqn import...`):

.
├── train.py                   # Main training loop
├── evaluate.py                # Evaluation and visualization script
├── requirements.txt           # Dependencies (see section 4)
├── agent/
│   ├── __init__.py            # (Empty file to make it a package)
│   ├── dqn.py                 # Neural Network class
│   └── replay_buffer.py       # Experience Replay Buffer
├── env/
│   ├── __init__.py            # (Empty file)
│   └── trading_env.py         # Custom Gym Environment
├── utils/
│   ├── __init__.py            # (Empty file)
│   └── data_preprocessing.py  # Data downloading and feature engineering
└── results/
    └── (Training logs and .pth models will be saved here)

4. PREREQUISITES & INSTALLATION
--------------------------------------------------------------------------------
This project requires Python 3.8+ and the following libraries:

- gymnasium
- torch
- pandas
- numpy
- yfinance
- ta (Technical Analysis Library)
- matplotlib
- scikit-learn

You can install them via pip:
pip install gymnasium torch pandas numpy yfinance ta matplotlib scikit-learn

5. HOW TO RUN
--------------------------------------------------------------------------------

STEP 1: TRAINING
Run the training script to download data, train the agent, and save the model.
The model will be saved to `results/dqn_spy.pth`.

    python train.py

*Note: Training 500 episodes takes approximately 10-15 minutes on a standard CPU.*

STEP 2: EVALUATION
Run the evaluation script to test the trained model on unseen data (the most
recent 20% of the dataset). This will generate performance plots and a
confusion matrix.

    python evaluate.py

6. CONFIGURATION
--------------------------------------------------------------------------------
You can adjust hyperparameters directly in `train.py`:
- `num_episodes`: Total training episodes (Default: 500)
- `lr`: Learning rate (Default: 1e-4)
- `gamma`: Discount factor (Default: 0.99)
- `batch_size`: Size of memory replay batch (Default: 64)

Environment settings in `env/trading_env.py`:
- `initial_balance`: Starting cash (Default: $10,000)
- `transaction_cost`: Commission per trade (Default: 0.1%)

7. METRICS & RESULTS
--------------------------------------------------------------------------------
The evaluation script outputs:
- **Trade Log:** A printed list of all executed Buys and Sells.
- **Performance Metrics:** Total Return, Alpha, Beta, and R-Squared.
- **Visualizations:**
  1. Portfolio Value vs. Buy & Hold Benchmark.
  2. Rolling Sharpe Ratio (Risk-adjusted return over time).
  3. Portfolio Drawdown (Risk analysis).
  4. Confusion Matrix (Agent decisions vs. Optimal hindsight decisions).

8. ACKNOWLEDGMENTS
--------------------------------------------------------------------------------
- Data provided by Yahoo Finance via `yfinance`.
- Technical indicators calculated using the `ta` library.
- Deep Learning framework provided by `PyTorch`.