import abc
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator, KAMAIndicator
from ta.volume import MFIIndicator
from ta.volatility import BollingerBands

# --- Clases para Almacenamiento de Datos (Más simples con Dataclasses) ---

@dataclass
class Portfolio:
    """Almacena el estado y el historial del portafolio."""
    initial_cash: float = 1_000_000
    cash: float = field(init=False)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def record_state(self, timestamp: Any, portfolio_value: float):
        """Guarda una instantánea del valor del portafolio en un momento dado."""
        self.history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })

    def get_history_df(self) -> pd.DataFrame:
        """Devuelve el historial del portafolio como un DataFrame."""
        return pd.DataFrame(self.history).set_index("timestamp")

@dataclass
class Transaction:
    """Representa una operación activa en el mercado."""
    entry_price: float
    n_shares: int
    transaction_type: str  # 'long' o 'short'
    stop_loss_price: float
    take_profit_price: float
    commission: float = 0.0125  # 1.25%

# --- Lógica de Estrategias (Ahora Vectorizadas) ---

class Strategy(abc.ABC):
    """Clase base abstracta para todas las estrategias de trading."""
    @abc.abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las señales de trading para todo el DataFrame.
        Devuelve el DataFrame con una nueva columna 'signal'.
        Señal: 1 para comprar, -1 para vender, 0 para mantener.
        """
        pass

class RSIStrategy(Strategy):
    def __init__(self, window: int = 14, buy_threshold: int = 30, sell_threshold: int = 70):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        rsi = RSIIndicator(data['Close'], window=self.window).rsi()
        data['signal'] = 0
        data.loc[rsi < self.buy_threshold, 'signal'] = 1
        data.loc[rsi > self.sell_threshold, 'signal'] = -1
        return data

class MAStrategy(Strategy):
    def __init__(self, window: int = 14):
        self.window = window

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['ma'] = KAMAIndicator(data['Close'], window=self.window).kama()
        data['signal'] = 0
        data.loc[data['Close'] > data['ma'], 'signal'] = 1
        data.loc[data['Close'] < data['ma'], 'signal'] = -1
        return data

class MFIStrategy(Strategy):
    def __init__(self, window: int = 14, buy_threshold: int = 30, sell_threshold: int = 70):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        mfi = MFIIndicator(
            high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=self.window
        ).money_flow_index()
        data['signal'] = 0
        data.loc[mfi < self.buy_threshold, 'signal'] = 1
        data.loc[mfi > self.sell_threshold, 'signal'] = -1
        return data

class BollingerBandsStrategy(Strategy):
    def __init__(self, window: int = 20, window_dev: int = 2):
        self.window = window
        self.window_dev = window_dev

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        indicator = BollingerBands(close=data["Close"], window=self.window, window_dev=self.window_dev)
        data['signal'] = 0
        data.loc[data['Close'] < indicator.bollinger_lband(), 'signal'] = 1
        data.loc[data['Close'] > indicator.bollinger_hband(), 'signal'] = -1
        return data

class CompoundStrategy(Strategy):
    """Combina múltiples estrategias. Opera solo si todas están de acuerdo."""
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        all_signals = pd.DataFrame(index=data.index)

        for i, strategy in enumerate(self.strategies):
            signal_df = strategy.calculate_signals(data)
            all_signals[f'signal_{i}'] = signal_df['signal']

        # Compra si todas las señales son 1, vende si todas son -1
        data['signal'] = 0
        data.loc[all_signals.eq(1).all(axis=1), 'signal'] = 1
        data.loc[all_signals.eq(-1).all(axis=1), 'signal'] = -1
        
        return data

# --- Motor de Backtesting ---

class Backtester:
    """Ejecuta una simulación de trading sobre datos históricos con señales pre-calculadas."""
    def __init__(self, data: pd.DataFrame, strategy: Strategy, n_shares: int, sl_tp_factors: tuple):
        """
        Args:
            data (pd.DataFrame): DataFrame con precios y la columna 'signal'.
            strategy (Strategy): Objeto de estrategia (no se usa para calcular, solo para referencia).
            n_shares (int): Número de acciones por operación.
            sl_tp_factors (tuple): Factores para (sl_long, tp_long, sl_short, tp_short).
                                   Ej: (0.95, 1.05, 1.05, 0.95)
        """
        self.data = data
        self.strategy = strategy
        self.n_shares = n_shares
        self.sl_l_factor, self.tp_l_factor, self.sl_s_factor, self.tp_s_factor = sl_tp_factors
        self.portfolio = Portfolio()
        self.active_operations: List[Transaction] = []

    def run(self):
        """Ejecuta el backtest."""
        for idx, row in self.data.iterrows():
            current_price = row['Close']
            
            # 1. Chequear y cerrar operaciones existentes (SL/TP)
            self._check_and_close_positions(current_price)

            # 2. Abrir nuevas operaciones según la señal
            signal = row['signal']
            # Solo abre una posición si no hay otra activa del mismo tipo
            if signal == 1 and not any(op.transaction_type == 'long' for op in self.active_operations):
                self._open_long(row)
            elif signal == -1 and not any(op.transaction_type == 'short' for op in self.active_operations):
                self._open_short(row)
            
            # 3. Calcular valor actual del portafolio y registrarlo
            assets_value = sum(op.n_shares * current_price for op in self.active_operations if op.transaction_type == 'long')
            # Para shorts, el "valor" es el beneficio/pérdida no realizado
            short_pnl = sum((op.entry_price - current_price) * op.n_shares for op in self.active_operations if op.transaction_type == 'short')
            
            portfolio_value = self.portfolio.cash + assets_value + short_pnl
            self.portfolio.record_state(idx, portfolio_value)

        return self.portfolio.get_history_df()

    def _check_and_close_positions(self, current_price: float):
        operations_to_remove = []
        for op in self.active_operations:
            closed = False
            if op.transaction_type == 'long':
                if current_price <= op.stop_loss_price:
                    self.portfolio.cash += op.stop_loss_price * op.n_shares * (1 - op.commission)
                    closed = True
                elif current_price >= op.take_profit_price:
                    self.portfolio.cash += op.take_profit_price * op.n_shares * (1 - op.commission)
                    closed = True
            
            elif op.transaction_type == 'short':
                if current_price >= op.stop_loss_price:
                    pnl = (op.entry_price - op.stop_loss_price) * op.n_shares
                    self.portfolio.cash += (op.entry_price * op.n_shares * (1-op.commission)) + pnl
                    closed = True
                elif current_price <= op.take_profit_price:
                    pnl = (op.entry_price - op.take_profit_price) * op.n_shares
                    self.portfolio.cash += (op.entry_price * op.n_shares * (1-op.commission)) + pnl
                    closed = True

            if closed:
                operations_to_remove.append(op)
        
        self.active_operations = [op for op in self.active_operations if op not in operations_to_remove]

    def _open_long(self, row: pd.Series):
        price = row['Close']
        cost = price * self.n_shares * (1 + Transaction.commission)
        if self.portfolio.cash >= cost:
            self.portfolio.cash -= cost
            self.active_operations.append(Transaction(
                entry_price=price,
                n_shares=self.n_shares,
                transaction_type='long',
                stop_loss_price=price * self.sl_l_factor,
                take_profit_price=price * self.tp_l_factor
            ))

    def _open_short(self, row: pd.Series):
        price = row['Close']
        # Al abrir un corto, el efectivo no se "gasta", se recibe (luego se devuelve)
        # Simplificaremos la contabilidad: el pnl se liquida al cierre.
        self.active_operations.append(Transaction(
            entry_price=price,
            n_shares=self.n_shares,
            transaction_type='short',
            stop_loss_price=price * self.sl_s_factor,
            take_profit_price=price * self.tp_s_factor
        ))

# --- Optimización y Cross-Validation ---

# Mapeo de nombres a clases para la optimización dinámica
STRATEGY_MAPPING = {
    'rsi': RSIStrategy,
    'ma': MAStrategy,
    'mfi': MFIStrategy,
    'b_b': BollingerBandsStrategy
}

def optimize_hyperparameters(param_grid: dict, strategies_to_combine: List[List[str]],
                             train_df: pd.DataFrame, validation_df: pd.DataFrame,
                             n_trials: int = 50):
    
    def objective(trial: optuna.Trial):
        # 1. Seleccionar un grupo de estrategias a combinar
        strat_names = trial.suggest_categorical('strategy_combination', [str(s) for s in strategies_to_combine])
        strat_names = eval(strat_names) # Convertir string de lista a lista

        # 2. Construir dinámicamente las instancias de estrategia con sus hiperparámetros
        strategy_instances = []
        for name in strat_names:
            params = {}
            # Busca los parámetros para esta estrategia en el param_grid
            for param, values in param_grid.get(name, {}).items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(f"{name}_{param}", values[0], values[1])
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(f"{name}_{param}", values[0], values[1])
            strategy_instances.append(STRATEGY_MAPPING[name](**params))
        
        # 3. Crear la estrategia compuesta
        compound_strategy = CompoundStrategy(strategies=strategy_instances)
        
        # 4. Sugerir otros parámetros del backtest
        n_s = trial.suggest_int('n_shares', param_grid['n_shares'][0], param_grid['n_shares'][1])
        sl_l = trial.suggest_float('sl_long_factor', 0.90, 0.99)
        tp_l = trial.suggest_float('tp_long_factor', 1.01, 1.10)

        # 5. Ejecutar backtest
        # Pre-calcular señales en todo el dataset de validación
        full_data = pd.concat([train_df, validation_df])
        data_with_signals = compound_strategy.calculate_signals(full_data)
        validation_data_with_signals = data_with_signals.loc[validation_df.index]
        
        backtester = Backtester(
            data=validation_data_with_signals,
            strategy=compound_strategy,
            n_shares=n_s,
            sl_tp_factors=(sl_l, tp_l, 1 + (1 - sl_l), 1 - (tp_l - 1)) # SL/TP simétricos para short
        )
        
        results = backtester.run()
        
        if results.empty:
            return -1.0 # Penalizar si no hay operaciones

        return calmar_ratio(results['portfolio_value'])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Mejores parámetros encontrados: {study.best_params}")
    return study.best_params

class CrossValidationBacktester:
    def __init__(self, full_data: pd.DataFrame, n_splits: int = 3):
        self.full_data = full_data
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

    def run_cv(self, strategy: Strategy, n_shares: int, sl_tp_factors: tuple):
        fold_results = {}
        
        for i, (train_index, test_index) in enumerate(self.tscv.split(self.full_data)):
            train_data = self.full_data.iloc[train_index]
            test_data = self.full_data.iloc[test_index]
            
            # El cálculo de señales debe usar datos de entrenamiento para evitar lookahead
            # aunque los indicadores como RSI se calculan sobre toda la serie disponible
            data_with_signals = strategy.calculate_signals(self.full_data)
            test_data_with_signals = data_with_signals.iloc[test_index]
            
            print(f"Ejecutando Fold {i+1}/{self.n_splits}...")
            
            backtester = Backtester(
                data=test_data_with_signals,
                strategy=strategy,
                n_shares=n_shares,
                sl_tp_factors=sl_tp_factors
            )
            
            portfolio_history = backtester.run()
            fold_results[i] = portfolio_history

        return fold_results

# --- Funciones de Utilidad y Visualización (Sin cambios mayores) ---

def calmar_ratio(portfolio_values: pd.Series, periods_per_year=8760):
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty:
        return 0.0

    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return 0.0

    mean_return = returns.mean()
    annualized_return = mean_return * periods_per_year
    
    peak = portfolio_values.cummax()
    drawdowns = (portfolio_values - peak) / peak
    max_drawdown = abs(drawdowns.min())

    if max_drawdown == 0:
        return np.inf

    return annualized_return / max_drawdown

def plot_backtesting_results(fold_results: dict):
    """Grafica los resultados del backtesting para cada fold."""
    fig = make_subplots(
        rows=len(fold_results),
        cols=1,
        subplot_titles=[f'Test Fold {k+1}' for k in fold_results.keys()]
    )

    for i, (key, df_value) in enumerate(fold_results.items()):
        row_index = i + 1
        fig.add_trace(go.Scatter(
            x=df_value.index,
            y=df_value['portfolio_value'],
            mode='lines',
            name=f'Fold {key+1}'
        ), row=row_index, col=1)

    fig.update_layout(
        title_text='Resultados del Backtesting por Fold',
        height=300 * len(fold_results),
        showlegend=False
    )
    fig.show()