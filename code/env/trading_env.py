import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for trading.
    Action space:
        0 = HOLD
        1 = BUY
        2 = SELL

    State includes:
        technical indicators + position flag + cash + asset holdings
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Trading state
        self.current_step = 0
        self.position = 0            # 0 = no position, 1 = long
        self.entry_price = 0
        self.cash = initial_balance
        self.shares = 0

        # ACTIONS: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

        # UPDATED: Use the normalized columns for the AI's input
        self.feature_columns = [
            "Open_norm", "High_norm", "Low_norm", "Close_norm", "Volume_norm",
            "RSI_norm", "MACD_norm", "MACD_signal_norm", "MACD_hist_norm",
            "SMA_20_norm", "SMA_50_norm", "BB_high_norm", "BB_low_norm", "Returns_norm",
            "VIX_norm", "TNX_norm", "ATR_norm", "OBV_norm"  # <--- Added here
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_columns) + 3,),
            dtype=np.float32
        )
        self.transaction_cost = 0.001


    # ========== Helper ==========
    def _get_state(self):
        """
        Construct full state vector.
        """
        row = self.df.iloc[self.current_step][self.feature_columns].values.astype(np.float32)

        # Normalize cash/shares relative to initial balance to keep inputs near 0-1 range
        scaled_cash = self.cash / self.initial_balance
        scaled_shares = (self.shares * self.df.iloc[self.current_step]["Close"]) / self.initial_balance

        state = np.concatenate([
            row,
            np.array([self.position, scaled_cash, scaled_shares], dtype=np.float32)
        ])
        return state


    # ========== Reset ==========
    def reset(self, seed=42, options=None):
        super().reset(seed=seed)

        # Random start index to prevent memorizing the dataset
        max_steps = len(self.df) - 500
        if max_steps > 0:
            self.current_step = np.random.randint(0, max_steps)
        else:
            self.current_step = 0

        self.position = 0
        self.entry_price = 0
        self.cash = self.initial_balance
        self.shares = 0

        return self._get_state(), {}


    # ========== Step ==========
    def step(self, action):
        price = float(self.df.iloc[self.current_step]["Close"])
        prev_value = self._get_portfolio_value()

        reward = 0

        # HOLD
        if action == 0:
            reward -= 0.0001

            # BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                cost = self.cash * self.transaction_cost
                self.shares = (self.cash - cost) / price
                self.cash = 0
            else:
                reward -= 0.1

        # SELL
        elif action == 2:
            if self.position == 1:
                proceeds = self.shares * price
                cost = proceeds * self.transaction_cost
                self.cash = proceeds - cost
                self.shares = 0
                self.position = 0
            else:
                reward -= 0.1

        # Advance step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Reward: Portfolio Value Change
        curr_value = self._get_portfolio_value()

        # SCALING
        reward += ((curr_value - prev_value) / prev_value) * 100

        return self._get_state(), float(reward), done, False, {}


    # ========== Portfolio Value ==========
    def _get_portfolio_value(self):
        price = float(self.df.iloc[self.current_step]["Close"])
        if self.position == 1:
            return self.cash + self.shares * price
        return self.cash
