import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from technical_analysis.utils.utils import *

if __name__ == "__main__":
        data = pd.read_csv("Notebook/files/Binance_BTCUSDT_1h.csv", header=1)
        data.rename(columns={'Volume BTC':'Volume'}, inplace=True)
        data.index = pd.to_datetime(data['Date'], format="ISO8601")
        data.sort_index(inplace=True)

        param_grid = {
                "rsi": {
                'window': [10, 50],
                'buy_threshold': [20, 40],
                'sell_threshold': [60, 80]
                },
                'n_shares': [1, 5]
        }

        strategies_to_combine = [['rsi']]

        train_size = int(len(data) * 0.7)
        train_df = data.iloc[:train_size]
        validation_df = data.iloc[train_size:]

        best_params = optimize_hyperparameters(
        param_grid,
        strategies_to_combine,
        train_df,
        validation_df,
        n_trials=50 
        )

        best_strat_names = eval(best_params['strategy_combination'])
        best_strategy_instances = []
        for name in best_strat_names:
                params = {k.split('_', 1)[1]: v for k, v in best_params.items() if k.startswith(name)}
                best_strategy_instances.append(STRATEGY_MAPPING[name](**params))

        final_strategy = CompoundStrategy(strategies=best_strategy_instances)
        final_n_shares = best_params['n_shares']
        final_sl_tp = (
        best_params['sl_long_factor'], 
        best_params['tp_long_factor'],
        1 + (1 - best_params['sl_long_factor']),
        1 - (best_params['tp_long_factor'] - 1)
        )


        cv_backtester = CrossValidationBacktester(full_data=data, n_splits=5)
        results = cv_backtester.run_cv(
        strategy=final_strategy, 
        n_shares=final_n_shares,
        sl_tp_factors=final_sl_tp
        )

        # Plot results
        plot_backtesting_results(results)
