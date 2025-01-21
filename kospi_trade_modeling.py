# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from prophet import Prophet
import requests
from bs4 import BeautifulSoup
from io import StringIO
from datetime import datetime
import math
import os
import warnings
import logging
warnings.filterwarnings("ignore")

sns.set()

# 상수 정의
TEST_SIZE = 0.2
RANDOM_SEED = 123
MOVING_AVERAGE_WINDOW = 252
SHORT_WINDOW = 28
LONG_WINDOW = 496
SIGNAL_THRESHOLD = 28
FORECAST_OUT = 84
EPOCH_NUM = 1000
LEARNING_RATE = 0.01
INPUT_DATA_COLUMN_CNT = 20
OUTPUT_DATA_COLUMN_CNT = 1
SEQ_LENGTH = 28
RNN_CELL_HIDDEN_DIM = 20
FORGET_BIAS = 1.0
NUM_STACKED_LAYERS = 1
KEEP_PROB = 1.0
MONTE_CARLO_ITERATIONS = 100
MONTE_CARLO_TIME_INTERVALS = 168
PROPHET_PERIODS = 496

# 데이터 로딩 및 전처리 함수
def load_and_preprocess_data(code, start_date, end_date):
    df = fdr.DataReader(code, start_date, end_date)
    df = df[df.Volume > 0]
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.set_index('Date')
    return df

# MDD 계산 함수
def get_mdd(x):
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    DT = x.index.values
    start_dt = pd.to_datetime(str(DT[peak_upper]))
    MDD_start = start_dt.strftime("%Y-%m-%d")
    end_dt = pd.to_datetime(str(DT[peak_lower]))
    MDD_end = end_dt.strftime("%Y-%m-%d")
    MDD_duration = np.busday_count(MDD_start, MDD_end)
    return MDD_start, MDD_end, (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper] * 100

# 기술적 지표 계산 함수
def calculate_technical_indicators(df, n=14):
    df['MMT'] = df['Close'].diff(n)
    df['전일대비등락'] = df['Close'] - df['Close'].shift(1)
    df['전일대비상승'] = 0
    df.loc[df['전일대비등락'] > 0, '전일대비상승'] = 1
    df['n일_상승_일수'] = df['전일대비상승'].rolling(window=n).sum()
    df['PSY'] = (df['n일_상승_일수'] / float(n)) * 100
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['BOL_UPPER'] = df['Close'].rolling(window=20).mean() + 4 * df['Close'].rolling(window=20).std()
    df['BOL_LOWER'] = df['Close'].rolling(window=20).mean() - 4 * df['Close'].rolling(window=20).std()
    return df

# 모멘텀 지표 시각화 함수
def plot_momentum(df, code, start_date, end_date, n=14):
    df = calculate_technical_indicators(df, n)
    col_name = 'MMT'
    st.write('*' * 100)
    st.write(f'{code}_Momentum : {round(df[col_name].iloc[-1], 2)}')
    st.write(f'{code}_Momentum Max Value : {round(df[col_name].max())}')
    st.write(f'{code}_Momentum Min Value : {round(df[col_name].min())}')
    st.write('*' * 100)
    fig, ax = plt.subplots(figsize=(12, 6))
    df[['Close', col_name]].plot(secondary_y=col_name, ax=ax, legend=True)
    ax.set_title('Momentum Index', color='white')
    ax.grid()
    ax.axhline(y=0, color='r', lw=2)
    st.pyplot(fig)

# 투자심리도 시각화 함수
def plot_psychological(df, code, start_date, end_date, n=14):
    df = calculate_technical_indicators(df, n)
    st.write('*' * 100)
    st.write(f'{code}_Psycological Index : {round(df["PSY"].iloc[-1], 2)}')
    st.write('*' * 100)
    fig, ax = plt.subplots(figsize=(12, 6))
    df[['Close', 'PSY']].plot(secondary_y=['PSY'], ax=ax, title='Psychological', legend=True)
    ax.set_title('Psychological Index', color='white')
    ax.grid()
    ax.axhline(y=75, color='r', lw=2)
    ax.axhline(y=25, color='r', lw=2)
    st.pyplot(fig)

# 볼린저 밴드 시각화 함수
def plot_bol_band_df(df, code, start_date, end_date, n=20, k=4):
    df = calculate_technical_indicators(df, n)
    fig, ax = plt.subplots(figsize=(12, 6))
    df[['Close', 'MA_20', 'BOL_UPPER', 'BOL_LOWER']].plot(color=list('brgg'), ax=ax, title='Bollinger Band')
    ax.fill_between(df.index, df.loc[:, 'BOL_UPPER'].values, df.loc[:, 'BOL_LOWER'].values, facecolor='g', alpha=0.2)
    st.pyplot(fig)

# 이동 변동성 시각화 함수
def plot_rolling_volatility(df):
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rol_Vol'] = df['Return'].rolling(window=MOVING_AVERAGE_WINDOW).mean() * math.sqrt(MOVING_AVERAGE_WINDOW)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    df[['Close', 'Rol_Vol', 'Return']].plot(ax=axes, subplots=True, style='g')
    st.pyplot(fig)

# 추세 시계열 매매 신호 시각화 함수
def plot_trend_trading_signals(df):
    df['short'] = df['Close'].rolling(center=False, window=SHORT_WINDOW).mean()
    df['long'] = df['Close'].rolling(center=False, window=LONG_WINDOW).mean()
    df['short-long'] = df['short'] - df['long']
    df['Regime'] = np.where(df['short-long'] > SIGNAL_THRESHOLD, 1, 0)
    df['Regime'] = np.where(df['short-long'] < SIGNAL_THRESHOLD, -1, df['Regime'])
    st.write(df['Regime'].value_counts())
    fig, ax = plt.subplots(figsize=(12, 6))
    df['Regime'].plot(ax=ax, lw=1.5, legend=True)
    ax.set_ylim([-1.1, 1.1])
    st.pyplot(fig)

# 이동 평균 전략 백테스팅 함수
def backtest_moving_average_strategy(df):
    df_backtest = pd.DataFrame()
    df_backtest['Close'] = df['Close']
    df_backtest['SHORT_MA'] = df_backtest['Close'].rolling(window=SHORT_WINDOW).mean()
    df_backtest['LONG_MA'] = df_backtest['Close'].rolling(window=LONG_WINDOW).mean()
    df_backtest['Signal'] = np.where(df_backtest['SHORT_MA'] > df_backtest['LONG_MA'], 1, 0)
    df_backtest['code_ret'] = np.log(df_backtest['Close'] / df_backtest['Close'].shift(1))
    df_backtest['str_ret'] = df_backtest['Signal'].shift(1) * df_backtest['code_ret']
    fig, ax = plt.subplots(figsize=(12, 6))
    df_backtest[['Close', 'SHORT_MA', 'LONG_MA']].plot(ax=ax)
    ax2 = ax.twinx()
    df_backtest['Signal'].plot(ax=ax2, color='r')
    st.pyplot(fig)
    st.write(df_backtest['Signal'].diff().value_counts())
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    df_backtest[['code_ret', 'str_ret']].cumsum().apply(np.exp).plot(ax=ax3)
    st.pyplot(fig2)
    st.write(df_backtest[['code_ret', 'str_ret']].cumsum().apply(np.exp).iloc[-1])

# 모멘텀 전략 타이밍 시각화 함수
def plot_momentum_strategy_timing(df, code):
    df['Short_MA'] = df['Close'].rolling(center=False, window=SHORT_WINDOW).mean()
    df['Long_MA'] = df['Close'].rolling(center=False, window=LONG_WINDOW).mean()
    df['diff'] = df['Short_MA'] - df['Long_MA']
    df = df[['Close', 'Short_MA', 'Long_MA', 'diff']]
    fig = plt.figure(figsize=(12, 6))
    price_chart = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    price_chart.plot(df.index, df['Close'], label='Close')
    price_chart.plot(df.index, df['Short_MA'], label='Short_MA')
    price_chart.plot(df.index, df['Long_MA'], label='Long_MA')
    price_chart.set_title(code)
    price_chart.legend(loc='best')
    price_chart.grid(c='y')
    plt.tight_layout()
    signal_chart = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    signal_chart.plot(df.index, df['diff'].fillna(0), color='g')
    signal_chart.axhline(y=0, linestyle='--', color='k')
    prev_val = 0
    for key, val in df['diff'].items():
        if val == 0:
            continue
        if val * prev_val < 0 and val > prev_val:
            signal_chart.annotate('B', xy=(key, df['diff'][key]), xytext=(10, -30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
        elif val * prev_val < 0 and val < prev_val:
            signal_chart.annotate('S', xy=(key, df['diff'][key]), xytext=(10, 30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
        prev_val = val
    st.pyplot(fig)

# 백테스팅 수익률 시각화 함수
def plot_backtesting_returns(df):
    bench_chg_code = df['Close'].pct_change()
    pct_ret = bench_chg_code.cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    pct_ret.plot(ax=ax)
    ax.fill_between(pct_ret.index, 0, pct_ret, where=pct_ret >= 0, facecolor='red', alpha=0.5)
    ax.fill_between(pct_ret.index, 0, pct_ret, where=pct_ret < 0, facecolor='blue', alpha=0.5)
    st.pyplot(fig)

# 백테스팅 로그 수익률 시각화 함수
def plot_backtesting_log_returns(df):
    bench_log_chg = np.log(df['Close'] / df['Close'].shift(1))
    log_ret = bench_log_chg.cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    log_ret.plot(ax=ax)
    ax.fill_between(log_ret.index, 0, log_ret, where=log_ret >= 0, facecolor='red', alpha=1)
    ax.fill_between(log_ret.index, 0, log_ret, where=log_ret < 0, facecolor='blue', alpha=1)
    st.pyplot(fig)

# 몬테카를로 시뮬레이션 함수
def monte_carlo_simulation(df):
    log_returns = np.log(1 + df['Close'].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    daily_returns = np.exp(drift + stdev * stats.norm.ppf(np.random.rand(MONTE_CARLO_TIME_INTERVALS, MONTE_CARLO_ITERATIONS)))
    S0 = df['Close'].iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for t in range(1, MONTE_CARLO_TIME_INTERVALS):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    price_list = pd.DataFrame(price_list)
    price_list['Close'] = price_list[0]
    close = pd.DataFrame(df['Close'])
    monte_carlo_forecast = pd.concat([close, price_list])
    monte_carlo = monte_carlo_forecast.iloc[:, :].values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monte_carlo)
    ax.axhline(y=S0, color='black', lw=4)
    st.pyplot(fig)
    st.write('*' * 100)
    st.write(f'Max Price {round(np.max(monte_carlo[-1]), 2)} / Min Price {round(np.min(monte_carlo[-1]), 2)} after {round(MONTE_CARLO_TIME_INTERVALS / 252, 1)} Years & Median Price {round(np.median(monte_carlo[-1]), 2)}')
    st.write('*' * 100)

# 데이터 전처리 함수
def preprocess_for_rnn(df, code):
    df.to_csv(f'{code}.csv')
    chart_data = pd.read_csv(f'./{code}.csv', usecols=[0, 1, 2, 3, 4, 5, 6])
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        chart_data[f'Close_ma{window}'] = chart_data['Close'].rolling(window).mean()
        chart_data[f'Volume_ma{window}'] = chart_data['Volume'].rolling(window).mean()
    training_data = chart_data.copy()
    training_data['Open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['Open_lastclose_ratio'].iloc[1:] = (training_data['Open'][1:].values - training_data['Close'][:-1].values) / training_data['Close'][:-1].values
    training_data['High_close_ratio'] = (training_data['High'].values - training_data['Close'].values) / training_data['Close'].values
    training_data['Low_close_ratio'] = (training_data['Low'].values - training_data['Close'].values) / training_data['Close'].values
    training_data['Close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['Close_lastclose_ratio'].iloc[1:] = (training_data['Close'][1:].values - training_data['Close'][:-1].values) / training_data['Close'][:-1].values
    training_data['Volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data['Volume_lastvolume_ratio'].iloc[1:] = (training_data['Volume'][1:].values - training_data['Volume'][:-1].values) / training_data['Volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
    for window in windows:
        training_data[f'Close_ma{window}_ratio'] = (training_data['Close'] - training_data[f'Close_ma{window}']) / training_data[f'Close_ma{window}']
        training_data[f'Volume_ma{window}_ratio'] = (training_data['Volume'] - training_data[f'Volume_ma{window}']) / training_data[f'Volume_ma{window}']
    training_data = training_data[['Date', 'Open', 'High', 'Low', 'Close_ma5', 'Close_ma10', 'Close_ma20', 'Close_ma60', 'Close_ma120',
                                   'Open_lastclose_ratio', 'High_close_ratio', 'Low_close_ratio', 'Close_lastclose_ratio',
                                   'Close', 'Volume', 'Volume_lastvolume_ratio', 'Volume_ma5', 'Volume_ma10', 'Volume_ma20', 'Volume_ma60', 'Volume_ma120']]
    training_data.to_csv(f'{code}_adjusted.csv')
    return training_data

# 데이터 정규화 함수
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# RNN 모델 학습 및 예측 함수
def train_and_predict_rnn(code):
    tf.random.set_seed(777)
    stock_file_name = f'{code}_adjusted.csv'
    encoding = 'euc-kr'
    names = ['Date', 'Open', 'High', 'Low', 'Close_ma5', 'Close_ma10', 'Close_ma20', 'Close_ma60', 'Close_ma120',
             'Open_lastclose_ratio', 'High_close_ratio', 'Low_close_ratio', 'Close_lastclose_ratio',
             'Close', 'Volume', 'Volume_lastvolume_ratio', 'Volume_ma5', 'Volume_ma10', 'Volume_ma20', 'Volume_ma60',
             'Volume_ma120']
    raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)
    raw_dataframe.drop('Date', axis=1, inplace=True)
    stock_info = raw_dataframe.values[120:].astype(np.float64)
    price = stock_info[:, :-7]
    norm_price = min_max_scaling(price)
    volume = stock_info[:, -7:]
    norm_volume = min_max_scaling(volume)
    x = np.concatenate((norm_price, norm_volume), axis=1)
    y = x[:, [4]]
    dataX = []
    dataY = []
    for i in range(0, len(y) - SEQ_LENGTH):
        _x = x[i: i + SEQ_LENGTH]
        _y = y[i + SEQ_LENGTH]
        dataX.append(_x)
        dataY.append(_y)
    train_size = int(len(dataY) * 0.8)
    testX = np.array(dataX[train_size:])
    testY = np.array(dataY[train_size:])
    trainX = np.array(dataX[:train_size])
    trainY = np.array(dataY[:train_size])
    model = Sequential()
    model.add(LSTM(units=RNN_CELL_HIDDEN_DIM, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, INPUT_DATA_COLUMN_CNT)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=RNN_CELL_HIDDEN_DIM, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=OUTPUT_DATA_COLUMN_CNT))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, epochs=50, batch_size=30, verbose=0)
    pred_y = model.predict(testX, verbose=0)
    fig, ax = plt.subplots()
    ax.plot(testY, color='red', label='real stock price')
    ax.plot(pred_y, color='blue', label='predicted stock price')
    ax.set_title('stock price prediction')
    ax.set_xlabel('time')
    ax.set_ylabel('stock price')
    ax.legend()
    st.pyplot(fig)
    recent_data = np.array([x[len(x) - SEQ_LENGTH:]])
    test_predict = reverse_min_max_scaling(price, pred_y[-1])
    st.write("Predict_stock price", test_predict[0])
    st.write('Last_price of tail(-1)', raw_dataframe[['Close']].iloc[-1])
    st.write('Difference of Price', test_predict[0] - raw_dataframe[['Close']].iloc[-1])
    st.write('Difference Ratio(%)', (test_predict[0] - raw_dataframe[['Close']].iloc[-1]) / test_predict[0] * 100)

# 머신러닝 회귀 모델 학습 및 예측 함수
def train_and_predict_ml_regression(df):
    df = df.dropna()
    df['Prediction'] = df[['Close']].shift(-FORECAST_OUT)
    X = np.array(df.drop(['Prediction'], axis=1))
    X = X[:-FORECAST_OUT]
    y = np.array(df['Prediction'])
    y = y[:-FORECAST_OUT]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    svm_confidence = svr_rbf.score(x_test, y_test)
    st.write("SVM Confidence: ", svm_confidence)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test)
    st.write("Linear Regression Confidence: ", lr_confidence)
    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-FORECAST_OUT:]
    lr_prediction = lr.predict(x_forecast)
    svm_prediction = svr_rbf.predict(x_forecast)
    df1 = pd.DataFrame(data=lr_prediction, columns=['lr_prediction'])
    df2 = pd.DataFrame(data=svm_prediction, columns=['svm_prediction'])
    df_ml = pd.concat([df1, df2], axis=1)
    df_ml.columns = ['lr_prediction', 'svm_prediction']
    st.write(df_ml.tail())
    st.write(df_ml.describe().T)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_ml.plot(ax=ax)
    st.pyplot(fig)
    st.write("\n")
    st.write("svm confidence : ", svm_confidence)
    st.write('---' * 100)
    st.write("lr confidence : ", lr_confidence)
    st.write('\n')

# Prophet 모델 학습 및 예측 함수
def train_and_predict_prophet(df, code):
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df.columns = ['ds', 'Open', 'High', 'Low', 'y']
    df = df[['ds', 'y']]
    m = Prophet(growth='linear',
                changepoints=['2020-02-25'],
                n_changepoints=28,
                changepoint_range=0.9,
                changepoint_prior_scale=0.5,
                seasonality_mode='additive',
                seasonality_prior_scale=10.0,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
                holidays=None,
                holidays_prior_scale=10.0,
                interval_width=0.8,
                mcmc_samples=0)
    m.fit(df)
    future = m.make_future_dataframe(periods=PROPHET_PERIODS)
    forecast = m.predict(future)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    plt.xticks(rotation=90)
    st.pyplot(fig1)
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

# 메인 함수
def main():
    try:
        st.title("KOSPI Fair Value Modeling")
        
        # 사이드바 파라미터 설정
        params = get_sidebar_parameters()
        
        # 데이터 로딩 및 전처리
        df = load_and_preprocess_data(params['code'], params['start_date'], params['end_date'])
        
        # MDD 계산 및 출력
        display_mdd(df)
        
        # 기술적 지표 분석
        display_technical_analysis(df, params)
        
        # 변동성 분석
        display_volatility_analysis(df)
        
        # 트레이딩 신호 분석
        display_trading_signals(df)
        
        # 백테스팅 분석
        display_backtesting_analysis(df, params)
        
        # 머신러닝 모델 분석
        display_ml_analysis(df, params)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}", exc_info=True)

def get_sidebar_parameters():
    """사이드바 파라미터를 가져오는 함수"""
    st.sidebar.header("Parameters")
    start_year = st.sidebar.number_input("Start Year", min_value=1900, max_value=2100, value=2000)
    start_month = st.sidebar.number_input("Start Month", min_value=1, max_value=12, value=1)
    start_day = st.sidebar.number_input("Start Day", min_value=1, max_value=31, value=2)
    end_year = st.sidebar.number_input("End Year", min_value=1900, max_value=2100, value=2025)
    end_month = st.sidebar.number_input("End Month", min_value=1, max_value=12, value=1)
    end_day = st.sidebar.number_input("End Day", min_value=1, max_value=31, value=9)
    code = st.sidebar.text_input("Stock Code", value='KS11')
    
    return {
        'start_date': datetime(start_year, start_month, start_day),
        'end_date': datetime(end_year, end_month, end_day),
        'code': code
    }

def display_mdd(df):
    """MDD 계산 및 출력"""
    try:
        mdd = get_mdd(df['Close'])
        st.write('*' * 100)
        st.write(f'Maximum Drawdown : {mdd:.2f}%')
        st.write('*' * 100)
    except Exception as e:
        st.error(f"Error calculating MDD: {str(e)}")
        logging.error(f"Error in MDD calculation: {str(e)}", exc_info=True)

def display_technical_analysis(df, params):
    """기술적 지표 분석 및 시각화"""
    try:
        st.header("Technical Indicators")
        plot_momentum(df.copy(), params['code'], params['start_date'], params['end_date'])
        plot_psychological(df.copy(), params['code'], params['start_date'], params['end_date'])
        plot_bol_band_df(df.copy(), params['code'], params['start_date'], params['end_date'])
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        logging.error(f"Error in technical analysis: {str(e)}", exc_info=True)

def display_volatility_analysis(df):
    """이동 변동성 시각화"""
    try:
        st.header("Rolling Volatility")
        plot_rolling_volatility(df.copy())
    except Exception as e:
        st.error(f"Error plotting rolling volatility: {str(e)}")
        logging.error(f"Error in rolling volatility analysis: {str(e)}", exc_info=True)

def display_trading_signals(df):
    """추세 시계열 매매 신호 시각화"""
    try:
        st.header("Trend Trading Signals")
        plot_trend_trading_signals(df.copy())
    except Exception as e:
        st.error(f"Error plotting trend trading signals: {str(e)}")
        logging.error(f"Error in trend trading signals analysis: {str(e)}", exc_info=True)

def display_backtesting_analysis(df, params):
    """이동 평균 전략 백테스팅"""
    try:
        st.header("Moving Average Strategy Backtesting")
        backtest_moving_average_strategy(df.copy())
    except Exception as e:
        st.error(f"Error backtesting moving average strategy: {str(e)}")
        logging.error(f"Error in moving average strategy backtesting: {str(e)}", exc_info=True)

def display_ml_analysis(df, params):
    """머신러닝 회귀 모델 학습 및 예측"""
    try:
        st.header("Machine Learning Regression")
        train_and_predict_ml_regression(df.copy())
    except Exception as e:
        st.error(f"Error training and predicting with ML regression: {str(e)}")
        logging.error(f"Error in machine learning regression analysis: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()
