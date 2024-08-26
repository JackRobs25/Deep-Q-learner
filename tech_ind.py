import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

def get_data(start, end, symbols, column_name="Adj Close", include_spy=True, data_folder="./data"):
    dates = pd.date_range(start, end)
    df1 = pd.DataFrame(index=dates)
    df2 = pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True, usecols=['Date', column_name])
    df2.rename(columns={column_name: "SPY"}, inplace=True)
    df1 = df1.join(df2, how='inner')
    for symbol in symbols:
        tmp_df = pd.read_csv(data_folder + '/'+ symbol + ".csv", index_col='Date', parse_dates=True, usecols=['Date', column_name])
        tmp_df.rename(columns={column_name: symbol}, inplace=True)
        df1 = df1.join(tmp_df, how='left', rsuffix='_'+symbol)
    if (not include_spy):
        df1.drop('SPY', axis=1, inplace=True)
    return df1

def simple_moving_average(data, window=20):
    return data.rolling(window=window).mean()


def bollinger_bands(data, window=9, num_std=2):
    sma = simple_moving_average(data, window=window)
    rolling_std = data.rolling(window=window).std()

    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)

    bollinger_band = sma

    result_df = pd.DataFrame({
        'Upper Band': upper_band,
        'Lower Band': lower_band,
        'Bollinger Band': bollinger_band
    }, index=data.index)  # Include the original index

    return result_df

def relative_strength_index(data):
    price_diff = data.diff(1)
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Initial RSI calculation (step one) using the first 14 days
    rs = avg_gain / avg_loss
    rsi_step_one = 100 - (100 / (1 + rs))

    # Smoothing step (step two)
    for i in range(14, len(rsi_step_one)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + loss.iloc[i]) / 14

    rs_smoothed = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs_smoothed))

    return rsi


def macd(data, symbol):
    df = data.copy()
    df['EMA-12'] = df[symbol].ewm(12).mean()
    df['EMA-26'] = df[symbol].ewm(26).mean()
    df['MACD'] = df['EMA-12'] - df['EMA-26']
    return df


