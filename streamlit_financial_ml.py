import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 페이지 설정
st.set_page_config(page_title="Financial Machine Learning App", layout="wide")
st.title("Financial Machine Learning Analysis")

# 사이드바 파라미터 설정
st.sidebar.header("모델 파라미터 설정")

# 주식 심볼 입력
ticker = st.sidebar.text_input("주식 심볼 입력", "AAPL")

# 날짜 범위 선택
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("시작일", datetime.now() - timedelta(days=365*3))
with col2:
    end_date = st.date_input("종료일", datetime.now())

# 모델 선택
model_type = st.sidebar.selectbox(
    "모델 선택",
    ["Random Forest", "선형 회귀", "LSTM"]
)

# Random Forest 파라미터
if model_type == "Random Forest":
    st.sidebar.subheader("Random Forest 파라미터")
    n_estimators = st.sidebar.slider(
        "트리 개수 (n_estimators)", 
        min_value=10, 
        max_value=500, 
        value=100,
        help="더 많은 트리를 사용하면 모델의 안정성이 향상되지만 학습 시간이 증가합니다."
    )
    
    max_depth = st.sidebar.slider(
        "최대 깊이 (max_depth)", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="트리의 최대 깊이를 제한하여 과적합을 방지합니다."
    )

# 선형 회귀 파라미터
elif model_type == "선형 회귀":
    st.sidebar.subheader("선형 회귀 파라미터")
    regression_type = st.sidebar.selectbox(
        "회귀 모델 유형",
        ["Linear", "Ridge", "Lasso"]
    )
    
    if regression_type in ["Ridge", "Lasso"]:
        alpha = st.sidebar.slider(
            "알파 (규제 강도)", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0,
            help="높은 값은 더 강한 규제를 의미합니다."
        )

# LSTM 파라미터
else:
    st.sidebar.subheader("LSTM 파라미터")
    sequence_length = st.sidebar.slider(
        "시퀀스 길이", 
        min_value=5, 
        max_value=60, 
        value=30,
        help="예측에 사용할 과거 데이터의 기간"
    )
    
    lstm_units = st.sidebar.slider(
        "LSTM 유닛 수", 
        min_value=32, 
        max_value=256, 
        value=128,
        help="모델의 복잡도를 결정합니다."
    )
    
    dropout_rate = st.sidebar.slider(
        "Dropout 비율", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.2,
        help="과적합 방지를 위한 dropout 비율"
    )
    
    learning_rate = st.sidebar.slider(
        "학습률", 
        min_value=0.0001, 
        max_value=0.01, 
        value=0.001,
        format="%.4f",
        help="모델의 학습 속도를 조절합니다."
    )

# 공통 파라미터
test_size = st.sidebar.slider(
    "테스트 데이터 비율", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.2,
    help="전체 데이터 중 테스트에 사용할 비율을 설정합니다."
)

# 데이터 다운로드 함수
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {e}")
        return None

# LSTM 데이터 준비 함수
def prepare_lstm_data(data, sequence_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# 메인 프로세스
if st.sidebar.button("분석 시작"):
    # 데이터 로드
    with st.spinner("주식 데이터를 다운로드 중입니다..."):
        stock_data = get_stock_data(ticker, start_date, end_date)
        
    if stock_data is not None:
        # 데이터 시각화
        st.subheader("주가 차트")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        ))
        st.plotly_chart(fig)

        if model_type in ["Random Forest", "선형 회귀"]:
            # 데이터 전처리
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data.dropna(inplace=True)
            
            X = stock_data[['Open', 'High', 'Low', 'Volume', 'Return']]
            y = stock_data['Close']
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )

            if model_type == "Random Forest":
                with st.spinner("Random Forest 모델을 학습 중입니다..."):
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
            else:  # 선형 회귀
                with st.spinner("선형 회귀 모델을 학습 중입니다..."):
                    if regression_type == "Linear":
                        model = LinearRegression()
                    elif regression_type == "Ridge":
                        model = Ridge(alpha=alpha)
                    else:  # Lasso
                        model = Lasso(alpha=alpha)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # 성능 평가
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            st.subheader("모델 성능")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col2:
                st.metric("R² Score", f"{r2:.2f}")

            if model_type == "Random Forest":
                # 특성 중요도 시각화
                st.subheader("특성 중요도")
                feature_importance = pd.DataFrame({
                    '특성': ['Open', 'High', 'Low', 'Volume', 'Return'],
                    '중요도': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('중요도', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=feature_importance['중요도'],
                    y=feature_importance['특성'],
                    orientation='h'
                ))
                fig.update_layout(title="특성 중요도 분석")
                st.plotly_chart(fig)

        else:  # LSTM
            with st.spinner("LSTM 모델을 학습 중입니다..."):
                # LSTM 데이터 준비
                X, y, scaler = prepare_lstm_data(stock_data, sequence_length)
                
                # 데이터 분할
                train_size = int(len(X) * (1 - test_size))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # LSTM 모델 구성
                model = Sequential([
                    LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, 1)),
                    Dropout(dropout_rate),
                    LSTM(lstm_units//2),
                    Dropout(dropout_rate),
                    Dense(1)
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='mse'
                )
                
                # 모델 학습
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0
                )
                
                # 예측
                predictions = model.predict(X_test)
                
                # 성능 평가
                mse = mean_squared_error(y_test, predictions)
                
                # 결과 표시
                st.subheader("LSTM 모델 성능")
                st.metric("Mean Squared Error", f"{mse:.6f}")
                
                # 학습 곡선 시각화
                st.subheader("학습 곡선")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Train Loss'
                ))
                fig.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss'
                ))
                fig.update_layout(
                    title="학습 과정",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig)
                
                # 예측 결과 시각화
                st.subheader("예측 결과")
                
                # 스케일 복원
                predictions_unscaled = scaler.inverse_transform(predictions)
                y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y_test_unscaled.flatten(),
                    name='실제 가격'
                ))
                fig.add_trace(go.Scatter(
                    y=predictions_unscaled.flatten(),
                    name='예측 가격'
                ))
                fig.update_layout(
                    title="가격 예측 결과",
                    xaxis_title="시간",
                    yaxis_title="가격"
                )
                st.plotly_chart(fig)

    else:
        st.error("데이터를 불러오는데 실패했습니다. 티커 심볼을 확인해주세요.") 
