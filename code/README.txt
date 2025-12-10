# Reinforcement Learning Trading Agent

## Project Structure

.
├── code/
│   ├── utils/
│   │   ├── data_preprocessing.py
│   │   └── __pycache__/
│   ├── agent/
│   │   ├── dqn.py
│   │   ├── replay_buffer.py
│   │   └── __pycache__/
│   ├── env/
│   │   ├── trading_env.py
│   │   └── __pycache__/
│   ├── results/
│   │   ├── performance_chart.png
│   │   ├── dqn_spy.pth
│   │   ├── episode_final_values.npy
│   │   ├── episode_rewards.npy
│   ├── train.py
│   ├── evaluate.py
│   └── README.txt
│
├── report/
│   ├── FE_slides.pdf
│   ├── FE_report.pdf
│   └── FE_video_link.txt
│
└── rubric/
    └── FE_rubric.pdf


## How to Install
pip install numpy pandas matplotlib torch gymnasium scikit-learn yfinance tqdm

## How to Train
cd code
python train.py

## How to Evaluate / Backtest
cd code
python evaluate.py


