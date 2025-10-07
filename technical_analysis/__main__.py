import pandas as pd
from utils import *

if __name__ == "__main__":
    data = pd.read_csv('aapl_1m_test.csv')
    split = int(data.shape[0]*0.80)
    data_train = data.iloc[:split]
    data_test = data.iloc[split:]
    trade = Trade(n_shares=5, sl=[0.95,1.05], tp=[1.05,0.95],
                      strategy=MA(window=12),  df_train=data_train)
    
    for idx, row in data_test.iterrows():
            trade.execute_strategy(row)
    print(trade.port.portfolio_value[-1])
