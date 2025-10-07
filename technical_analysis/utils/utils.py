import abc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import List
from ta.momentum import RSIIndicator, KAMAIndicator
from ta.volume import MFIIndicator
from sklearn.model_selection import TimeSeriesSplit
from ta.volatility import BollingerBands
import optuna
import itertools

from time import time

from performance_metrics.metrics import calmar_ratio

def strats_combinations(strats:List):
    combinations = []
    for r in range(1, len(strats)+1):
        combinations.extend([list(x) for x in itertools.combinations(iterable=strats, r=r)])
    return combinations

def add_instance(df:pd.DataFrame, new_instance:pd.Series):
    df = df.copy()
    new_instance = pd.DataFrame(new_instance).T
    df = pd.concat([df, new_instance], axis=0)

    return df
  
class Portfolio:
    def __init__(self):
        self.cash = 1_000_000
        self.active_operations = []
        self.portfolio_value = []
        self.asset_values = []
        self.margin_acount_value = []

class Transaction:
    def __init__(self, row):
        self.bought_at = row.Close
        self._com = 1.25/100  

class LongTransaction(Transaction):
    def __init__(self, row,sl: float, tp: float):
        super().__init__(row)
        self.type_transaction = 'long'
        self.stop_loss = row.Close*sl
        self.take_profit = row.Close*tp
    
class ShortTransaction(Transaction):
    def __init__(self, row, sl:float, tp:float):
        super().__init__(row)
        self.type_transaction = 'short'
        self.stop_loss = row.Close*sl
        self.take_profit = row.Close*tp
        self.margin_acount = row.Close * 0.20

class Trade:
    def __init__(self, n_shares, sl:List, tp:List, strategy:object,
                 df_train=None):
        '''
        sl(spot loss) y tp(take profit) seran lista, donde el primera valor
        correspondera a la operaciones Long y el segundo a las operaciones
        Short
        '''
        self.port = Portfolio()
        self.n_shares = n_shares

        self.sl_l = sl[0]
        self.tp_l = tp[0]

        self.sl_s = sl[1]
        self.tp_s = tp[1]
        self.strategy = strategy
        self.df_train = df_train

        self.strategy.df_train = self.df_train

        
    def __check_stop_take(self, row):
        active_op_temp = []
        for operation in self.port.active_operations:
            if operation.type_transaction == 'long':
                if operation.stop_loss > row.Close:
                    self.port.cash += (operation.stop_loss * self.n_shares)*(1-operation._com)
                elif operation.take_profit <  row.Close:
                    self.port.cash += (operation.take_profit * self.n_shares)*(1-operation._com)
                else:
                    active_op_temp.append(operation)

            elif operation.type_transaction == 'short':
                if operation.stop_loss < row.Close:
                    self.port.cash -= (operation.stop_loss * self.n_shares)*(1+operation._com) \
                                     -((operation.margin_acount - abs(operation.stop_loss - operation.bought_at))*self.n_shares)
                elif operation.take_profit >  row.Close:
                    self.port.cash -= (operation.stop_loss * self.n_shares)*(1+operation._com) \
                                      -((operation.margin_acount + abs(operation.take_profit - operation.bought_at))*self.n_shares)                   
                else:
                    active_op_temp.append(operation)

        self.port.active_operations = active_op_temp
        self.port.margin_acount_value = np.sum([operation.margin_acount
                                                    for operation 
                                                    in self.port.active_operations 
                                                    if operation.type_transaction == 'short'])

    def __check_capital(self, row):
        transaction = Transaction(row)
        
        if self.port.cash < (row.Close * (self.n_shares*(1+transaction._com))):
            
            self.port.asset_values = sum([self.n_shares*row.Close
                                            for operation in self.port.active_operations])
            
            self.port.portfolio_value.append(self.port.cash + self.port.asset_values + self.port.margin_acount_value)

    def execute_strategy(self, row):
        self.__check_stop_take(row)
        self.__check_capital(row)

        self.df_train = add_instance(self.df_train, row)
        start = time()
        self.strategy.df_train = self.df_train 
        signal = self.strategy.get_signals(row)

        if signal == 'buy':
            transaction = LongTransaction(row, sl=self.sl_l,tp=self.tp_l)
            self.port.active_operations.append(transaction)
            self.port.cash -= row.Close * (self.n_shares*(1+transaction._com))

        elif signal == 'sell':
            transaction = ShortTransaction(row,sl=self.sl_s,tp=self.tp_s)
            self.port.active_operations.append(transaction)
            self.port.cash += (row.Close * self.n_shares)*(1-transaction._com)\
                             -(transaction.margin_acount * self.n_shares)
            self.port.margin_acount_value += (transaction.margin_acount * self.n_shares)

        self.port.asset_values = sum([self.n_shares*row.Close
                                      for operation in self.port.active_operations])
        
        self.port.portfolio_value.append(self.port.cash + self.port.asset_values + self.port.margin_acount_value)

        end = time()
        print(f"Tiempo de ejecución: {end-start} segundos")

class Strategy(abc.ABC):
    def __init__(self, df_train):
        '''
        >   df_train representara la data con la cual se calculara
            o se entrenada tu estrategia
       
        >   Ejemplo de ejecucion de estrategias:
            trade =  Trade(n_shares=2, sl=[0.95,1.05], tp=[1.05,0.95 ])
            for idx, row in data.iterrows():
                trade.execute_strategy(RSI(),row)

            trade.port.portfolio_value
        '''

        self.df_train = df_train

    @abc.abstractclassmethod
    def get_signals(row, *args):
        pass


        
## Apartir de aqui se crearan las clases que representaran los indicadores


class RSI(Strategy):
    def __init__(self, df_train=None,
                 window=14,
                 buy_threshold=30,
                 sell_threshold = 70):
        
        super().__init__(df_train)

        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold   

    def get_signals(self, row):
        """
        Determines the trading signal based on the latest Relative Strength Index (RSI) value computed
        on a subset of the training dataframe.
        This function calculates the RSI for the adjusted close prices in the training dataframe (excluding
        the last 'window' number of observations) using the configured window size. It then checks the most
        recent RSI value: if it is below the defined buy threshold, the function returns a 'buy' signal;
        if it is above the defined sell threshold, it returns a 'sell' signal; otherwise, it returns '-' to indicate
        no clear signal.
        Parameters:
            row (Any): A row of data passed to the function. Note that this parameter is not utilized in the current 
                       implementation but is included for future extensibility or compatibility with similar methods.
        Returns:
            str: A signal string - 'buy', 'sell', or '-' based on the computed RSI value against the set thresholds.
        """

        self.df_train['rsi'] = RSIIndicator(self.df_train.Close[:-self.window],
                                            window=self.window).rsi()

        if self.df_train.rsi.iloc[-1] < self.buy_threshold:
            return 'buy'
          
        elif self.df_train.rsi.iloc[-1] > self.sell_threshold:
            return 'sell'
        else:
            return '-'
        
class MA(Strategy):
    def __init__(self, df_train=None,
                 window=14):
        super().__init__(df_train)

        self.window = window

    def get_signals(self, row):
        self.df_train['ma'] = KAMAIndicator(self.df_train.Close[:-self.window],
                                            window=self.window).kama()

        if self.df_train.Close.iloc[-1] > self.df_train.ma.iloc[-1]:
            return 'buy'
          
        elif self.df_train.Close.iloc[-1] < self.df_train.ma.iloc[-1]:
            return 'sell'
        else:
            return '-'

class MFI(Strategy):
    def __init__(self, df_train=None,
                 window=14,
                 buy_threshold=30,
                 sell_threshold = 70):
        
        super().__init__(df_train)

        self.window = window
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_signals(self, row):

        self.df_train['mfi'] = MFIIndicator(self.df_train.High[:-self.window],
                                            self.df_train.Low[:-self.window],
                                            self.df_train.Close[:-self.window],
                                            self.df_train.Volume[:-self.window],
                                            window=self.window).money_flow_index() 

        if self.df_train.mfi.iloc[-1] < self.buy_threshold:
            return 'buy'
          
        elif self.df_train.mfi.iloc[-1] > self.sell_threshold:
            return 'sell'
        else:
            return '-'
        
class B_B(Strategy):
    def __init__(self, df_train=None,
                 window=14,
                 window_dev=2):
        
        super().__init__(df_train)
        self.window = window
        self.window_dev = window_dev

    def get_signals(self, row):
        bbands = BollingerBands(self.df_train.Close[:-self.window],
                                            window=self.window,
                                            window_dev=self.window_dev)
        
        self.df_train['bband_h'] = bbands.bollinger_hband()
        self.df_train['bband_l'] = bbands.bollinger_lband()

        if self.df_train.Close.iloc[-1] < self.df_train['bband_l'].iloc[-1]:
            return 'buy'
          
        if self.df_train.Close.iloc[-1] > self.df_train['bband_h'].iloc[-1]:
            return 'sell'
        else:
            return '-'
    
        
class Compoud_Strategies(Strategy):
    def __init__(self, strategies: List, df_train=None):
        '''
        En estrategias agregaras una tupla que contenga el conjunto
        de estrategias que se deseen probar
        '''
        super().__init__(df_train)
        self.strategies = strategies

    def get_signals(self,row):
        decision_strat = []

        for strat in self.strategies:
            strat.df_train = self.df_train
            decision_strat.append(strat.get_signals(row))

        if np.all(np.array(decision_strat) == 'buy'):
            return 'buy'
        elif np.all(np.array(decision_strat) == 'sell'):
            return 'sell'
        else:
            return '-'
        

def optimize_strats(param_grid:dict, strategies:List[List],
                    train_df:pd.DataFrame, validation_df:pd.DataFrame):
    '''
    Ejemplo de optimizacion:
    param_grid = {"rsi": {'window':[14,30]},
                 "mcd": {'window':[14,30]},
                 'n_shares':[2,6]}
    strategies = [['rsi','mcd'], ['rsi'], ['mcd']]
    optimize_strats(param_grid=param_grid,
                    strategies=strategies
    '''
    def objective(trial):
        strategies_instance = {}
        n_s = trial.suggest_int('n_shares',param_grid['n_shares'][0],
                                           param_grid['n_shares'][1])  

        if 'rsi' in param_grid:
            rsi_window = trial.suggest_int('rsi_window',param_grid['rsi']['window'][0],
                                       param_grid['rsi']['window'][1])
            
            buy_rsi = trial.suggest_int('rsi_buy', param_grid['rsi']['buy_threshold'][0],
                                                  param_grid['rsi']['buy_threshold'][1])
            
            sell_rsi = trial.suggest_int('rsi_sell', param_grid['rsi']['sell_threshold'][0],
                                                  param_grid['rsi']['sell_threshold'][1])

            strategies_instance['rsi'] = RSI(window=rsi_window,
                                             buy_threshold=buy_rsi,
                                             sell_threshold=sell_rsi)

        if 'mfi' in param_grid:
            mfi_window = trial.suggest_int('mfi_window',param_grid['mfi']['window'][0],
                                       param_grid['mfi']['window'][1])
            
            buy_mfi = trial.suggest_int('mfi_buy', param_grid['mfi']['buy_threshold'][0],
                                                  param_grid['mfi']['buy_threshold'][1])
            
            sell_mfi = trial.suggest_int('mfi_sell', param_grid['mfi']['sell_threshold'][0],
                                                  param_grid['mfi']['sell_threshold'][1])
            
            strategies_instance['mfi'] = MFI(window=mfi_window,
                                             buy_threshold=buy_mfi,
                                             sell_threshold=sell_mfi)
            
        if 'b_b' in param_grid:
            bb_window = trial.suggest_int('bb_window',param_grid['b_b']['window'][0],
                                       param_grid['b_b']['window'][1])
            
            bb_window_dev = trial.suggest_int('bb_window_dev',param_grid['b_b']['window_dev'][0],
                                       param_grid['b_b']['window_dev'][1])
                        
            strategies_instance['b_b'] = B_B(window=mfi_window,
                                             window_dev=bb_window_dev)

        if 'ma' in param_grid:
            ma_window = trial.suggest_int('ma_window',param_grid['ma']['window'][0],
                                       param_grid['ma']['window'][1])
            
            strategies_instance['ma'] = MA(window=ma_window)

        strat = trial.suggest_categorical('strategy',choices=strategies)

        compound_strats = Compoud_Strategies(strategies=[strategies_instance[i]
                                                            for i in strat])

        trade = Trade(n_shares=n_s, sl=[0.95,1.05], tp=[1.05,0.95],
                      strategy=compound_strats,  df_train=train_df)
        
        for idx, row in validation_df.iterrows():
            trade.execute_strategy(row)

        calmar_ratio_port = calmar_ratio(pd.Series(trade.port.portfolio_value))
        return calmar_ratio_port
    
    study = optuna.study.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(f"Best params are {study.best_params}")
    return study.best_params
    
class BackTest():
    def __init__(self,
                 df_train,
                 df_test,
                 n_split=3):
        
        self.df_train = df_train
        self.df_test = df_test
        self.n_split = n_split

    def back_test(self, trade:object):
        self.trade = trade
        full_data = pd.concat([self.df_train, self.df_test], axis=0)
        tscv = TimeSeriesSplit(n_splits=self.n_split)
        back_test_fold = {}
        back_test_fold_pvalue = {}
        for i, (train_index,test_index) in enumerate(tscv.split(full_data)):
            self.trade.df_train = full_data.iloc[train_index]

            for _, row in full_data.iloc[test_index].iterrows():
                self.trade.execute_strategy(row)

            back_test_fold[i] = self.trade.port.portfolio_value[-1]
            
            back_test_fold_pvalue[i] = [self.trade.port.portfolio_value,
                                            pd.to_datetime(full_data.iloc[test_index].Date)]
            self.trade = trade
    
        return back_test_fold, back_test_fold_pvalue


def plot_backtesting(port_value: dict):
    """
    Grafica los resultados del backtesting para cada fold.
    """
    fig = make_subplots(rows=len(port_value),
                        cols=1,
                        subplot_titles=[f'Test Fold {k}' for k in port_value.keys()])

    for i, (key, value) in enumerate(port_value.items()):
        row_index = i + 1

        fig.add_trace(go.Scatter(
            x=value[1],  # Fechas o timestamps
            y=value[0],  # Valores del portafolio
            mode='lines',
            line={'color': 'firebrick', 'dash': 'dash'},
            name=f'Fold {key}' # Usamos la llave real para el nombre
        ), row=row_index, col=1)

    fig.update_layout(
        title_text='Resultados del Backtesting por Fold',
        height=250 * len(port_value), # Ajusta la altura dinámicamente
        showlegend=False
    )
    fig.show()

def candle_charts(*args:pd.DataFrame,**kwargs):
    fig = make_subplots(rows=len(args), cols=1)
    for i, d in enumerate(args):
        try:
            fig.add_trace(go.Candlestick(x=d.Date,
                                        open=d.Open,
                                        high=d.High,
                                        low=d.Low,
                                        close=d.Close,
                                        row=i+1,
                                        col=1,
                                        increasing_line_color= 'cyan',
                                        decreasing_line_color= 'red'))
        except:
            fig.add_trace(go.Candlestick(x=d.Datetime,
                                        open=d.Open,
                                        high=d.High,
                                        low=d.Low,
                                        close=d.Close,
                                        row=i+1,
                                        col=1,
                                        increasing_line_color= 'cyan',
                                        decreasing_line_color= 'red'))
            
    fig.update_layout(title=f'Time series data')
    fig.show()

def plot_backtesting(port_value:dict):
    fig = make_subplots(row=len(port_value),
                        col=1)
    for  k, v in port_value.items():
        fig.add_trace(go.Scatter(x=v[1],
                                 y=v[0],
                                line={'color':'firebrick',
                                    'dash':'dash'}),
                                    row=k+1,
                                    col=1,
                                    name=f'test fold {k}')
        
    fig.update_layout(title='Backtest folds')
    fig.show()