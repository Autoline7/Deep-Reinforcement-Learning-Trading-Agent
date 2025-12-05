import torch
from utils.data_preprocessing import load_and_prepare_data, train_test_split
from env.trading_env import TradingEnv
from agent.dqn import DQN

# Load data
df = load_and_prepare_data("data/SPY.csv")
train_df, test_df = train_test_split(df)

# Create env
env = TradingEnv(train_df)
state, _ = env.reset()

# Convert to tensor
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# Create DQN
model = DQN(input_dim=len(state), output_dim=3)

# Forward pass
q_values = model(state_tensor)
print("Q-values:", q_values)
print("Shape:", q_values.shape)
