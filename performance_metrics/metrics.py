import numpy as np
import pandas as pd

def sharpe_ratio(porfolio_value:list[float], rf:float=0.035) -> float: # Ramon
    """
    Calculate the annualized Sharpe Ratio for a given portfolio.
    This function computes the Sharpe Ratio by calculating the portfolio's periodic returns,
    annualizing both the mean return and the standard deviation (volatility), and then dividing
    the excess return (mean return minus risk-free rate) by the annualized volatility.
    Parameters:
        porfolio_value (list[float]): A list of portfolio values over time.
        rf (float, optional): The annualized risk-free rate. Default is 0.035.
    Returns:
        float: The Sharpe Ratio. If an error occurs during calculation, returns NaN.
    """
    try:
        port_value = pd.Series(np.array(porfolio_value))
        ret_port = port_value.pct_change().dropna()
        vol_port = ret_port.std()*np.sqrt(8760)
        gain_port = ret_port.mean() * 8760
        return (gain_port - rf) / vol_port
    except Exception as e:
        print("Error calculating Sharpe Ratio:", e)
        return np.nan

def sortino_ratio(porfolio_value:list[float], rf:float=0.035) -> float: # Mariana
    """
    Calculate the Sortino ratio for a given portfolio.
    The Sortino ratio measures the risk-adjusted return of an investment asset, portfolio,
    or strategy by comparing the excess return over a risk-free rate to the downside deviation
    (annualized) of returns.
    Parameters:
        porfolio_value (list[float]): A list of portfolio values at different time intervals 
            (typically chronological). The function computes returns based on these values.
        rf (float, optional): The annualized risk-free rate of return. Default value is 0.035.
    Returns:
        float: The Sortino ratio, calculated as the excess return (annualized) divided by the 
        annualized negative volatility of the portfolio returns. If an error occurs during 
        computation, the function prints an error message and returns numpy.nan.
    Notes:
        - The function computes periodic returns using percentage change and drops the 
          initial missing value.
        - Downside deviation is calculated only from negative returns and scaled by the 
          square root of 8760 (assumed to represent the number of periods in a year based on 
          hourly data).
        - Be sure that the input list is not empty and that the data frequency corresponds 
          to the annualization factor used in the function.
    Example:
        >>> portfolio = [100, 102, 101, 105, 107]
        >>> sortino = sortino_ratio(portfolio)
        >>> print(sortino)
    """
    try:
        port_value = pd.Series(np.array(porfolio_value))
        ret_port = port_value.pct_change().dropna()
        vol_port_neg = ret_port[ret_port<0].std()*np.sqrt(8760)
        gain_port = ret_port.mean() * 8760
        return (gain_port - rf) / vol_port_neg
    except Exception as e:
        print("Error calculating Sharpe Ratio:", e)
        return np.nan    
    
def maximum_drawdown(portfolio_value:list[float])->float: # Ramon
    """
    Calculate the maximum drawdown of a portfolio based on its historical values.
    The function computes the drawdown by first determining the percentage change
    in the portfolio values, then taking a cumulative sum of these percentage changes,
    and finally returning the minimum value of this cumulative series, which represents
    the maximum drawdown. In the event of an error during the calculation, the function
    prints an error message and returns NaN.
    Parameters:
        portfolio_value (list[float]): A list of portfolio values over time.
    Returns:
        float: The maximum drawdown value (typically negative) computed from the cumulative
        percentage returns of the portfolio. If an error occurs during computation, returns NaN.
    """
    try:
        port_value = pd.Series(np.array(portfolio_value))
        ret_port = port_value.pct_change().dropna()
        return ret_port.cumsum().min()
    except Exception as e:
        print("Error calculating Sharpe Ratio:", e)
        return np.nan 
    
def win_rate(portfolio_value:list[float]) -> float: # Mariana
    """
    Calculates the average positive daily return (win rate) of a portfolio.
    This function converts a list of portfolio values into a pandas Series,
    computes the daily percentage change, filters for positive returns (wins),
    and then returns the mean of these positive returns. If any error
    occurs during this process, an error message is printed and NaN is returned.
    Parameters:
        portfolio_value (list[float]): A list of float values representing the 
                                       portfolio's value over time.
    Returns:
        float: The average positive hourly return (win rate) based on the portfolio's returns.
               If an error occurs during calculation, NaN is returned.
    """

    try:
        port_value = pd.Series(np.array(portfolio_value))
        ret_port = port_value.pct_change().dropna()
        return (ret_port>0).mean()
    except Exception as e:
        print("Error calculating Sharpe Ratio:", e)
        return np.nan
    
def calmar_ratio(valores_portafolio, periodos_anuales=8760): # General
    """
    Calculate the Calmar Ratio for a given portfolio series.
    This function computes the Calmar Ratio as the ratio of the annualized return to the maximum drawdown observed
    in the portfolio values. The annualized return is derived from the mean percentage change scaled by the number of periods in a year.
    The maximum drawdown is calculated as the largest drop from a historical peak in the portfolio. If the maximum drawdown is zero,
    indicating no losses, the function returns infinity.
    Parameters:
        valores_portafolio (pd.Series): A pandas Series representing the portfolio values over time.
        periodos_anuales (int, optional): Number of periods per year to annualize the return. Default is 8760.
    Returns:
        float: The Calmar Ratio, which is the annualized return divided by the maximum drawdown. Returns np.inf if the maximum drawdown is zero.
    Raises:
        TypeError: If 'valores_portafolio' is not a pandas Series.
    """
    if not isinstance(valores_portafolio, pd.Series):
        raise TypeError("El input debe ser una serie de pandas (pd.Series).")

    rendimientos = valores_portafolio.pct_change().dropna()

    rendimiento_medio = rendimientos.mean()
    rendimiento_anualizado = rendimiento_medio * periodos_anuales

    pico_anterior = valores_portafolio.cummax()
    drawdowns = (valores_portafolio - pico_anterior) / pico_anterior
    max_drawdown = abs(drawdowns.min())

    if max_drawdown == 0:
        return np.inf

    calmar_ratio = rendimiento_anualizado / max_drawdown

    return calmar_ratio