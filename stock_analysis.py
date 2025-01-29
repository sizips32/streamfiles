import os
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool, SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import talib

warnings.filterwarnings('ignore')

# .env 파일 로드
load_dotenv()

# Streamlit Secrets에서 API 키 가져오기
def get_api_keys():
    if st.runtime.exists():
        # Streamlit Cloud에서 실행 중일 때
        return {
            'SERPER_API_KEY': st.secrets['SERPER_API_KEY'],
            'OPENAI_API_KEY': st.secrets['OPENAI_API_KEY'],
            'ANTHROPIC_API_KEY': st.secrets['ANTHROPIC_API_KEY']
        }
    else:
        # 로컬에서 실행 중일 때
        return {
            'SERPER_API_KEY': os.getenv('SERPER_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
        }

# API 키 설정
api_keys = get_api_keys()
os.environ['SERPER_API_KEY'] = api_keys['SERPER_API_KEY']
os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']

# 웹 검색 툴 설정
search_tool = SerperDevTool()

class FinancialAnalyzer:
    @staticmethod
    def format_number(number):
        """숫자 포맷팅"""
        if number is None or pd.isna(number):
            return "N/A"
        return f"{number:,.0f}"

    @staticmethod
    def calculate_growth_rate(current, previous):
        """성장률 계산"""
        if previous and current and previous != 0:
            return (current - previous) / abs(previous) * 100
        return None

    @staticmethod
    def format_financial_summary(financials):
        """재무 요약 포맷팅"""
        summary = {}
        for date, data in financials.items():
            date_str = date.strftime('%Y-%m-%d')
            summary[date_str] = {
                "총수익": FinancialAnalyzer.format_number(data.get('TotalRevenue')),
                "영업이익": FinancialAnalyzer.format_number(data.get('OperatingIncome')),
                "순이익": FinancialAnalyzer.format_number(data.get('NetIncome')),
                "EBITDA": FinancialAnalyzer.format_number(data.get('EBITDA')),
                "EPS(희석)": f"${data.get('DilutedEPS'):.2f}" if pd.notna(data.get('DilutedEPS')) else "N/A"
            }
        return summary

    def get_historical_data(self, ticker, period):
        """주가 데이터 가져오기"""
        end_date = datetime.now()
        period_days = {
            "1개월": 30,
            "3개월": 90,
            "6개월": 180,
            "1년": 365
        }
        start_date = end_date - timedelta(days=period_days.get(period, 30))
        
        stock = yf.Ticker(ticker)
        try:
            historical_prices = stock.history(start=start_date, end=end_date, interval='1d')
            if historical_prices.empty:
                return None, None, "분석 불가: 최근 데이터가 없습니다."
            
            return historical_prices, stock, None
        except Exception as e:
            return None, None, f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}"

    def analyze_financials(self, stock):
        """재무제표 분석"""
        try:
            annual_financials = stock.get_financials()
            quarterly_financials = stock.get_financials(freq="quarterly")
            balance_sheet = stock.get_balance_sheet()
            
            # 주요 재무 지표 (연간)
            latest_financials = {
                'revenue': annual_financials.loc['TotalRevenue', annual_financials.columns[0]],
                'cost_of_revenue': annual_financials.loc['CostOfRevenue', annual_financials.columns[0]],
                'gross_profit': annual_financials.loc['GrossProfit', annual_financials.columns[0]],
                'operating_income': annual_financials.loc['OperatingIncome', annual_financials.columns[0]],
                'net_income': annual_financials.loc['NetIncome', annual_financials.columns[0]],
                'ebitda': annual_financials.loc['EBITDA', annual_financials.columns[0]],
                'total_assets': balance_sheet.loc['TotalAssets', balance_sheet.columns[0]]
            }
            
            return latest_financials, None
        except Exception as e:
            return None, f"재무 데이터를 가져오는 중 오류가 발생했습니다: {str(e)}"

class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(data, periods=14):
        """RSI 계산"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
        """MACD 계산"""
        short_ema = data.ewm(span=short_period, adjust=False).mean()
        long_ema = data.ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20):
        """볼린저 밴드 계산"""
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return upper, middle, lower
    
    @staticmethod
    def calculate_stochastic(data, k_period=14, d_period=3):
        """스토캐스틱 계산"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def calculate_obv(data):
        """OBV(On Balance Volume) 계산"""
        obv = pd.Series(index=data.index)
        obv.iloc[0] = 0
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

class MarketDataAnalyzer:
    def __init__(self, symbol, period='6mo'):
        self.symbol = symbol
        self.period = period
        self.data = self._fetch_data()
        self.technical = TechnicalAnalysis()
        
    def _fetch_data(self):
        """야후 파이낸스에서 데이터 가져오기"""
        stock = yf.Ticker(self.symbol)
        df = stock.history(period=self.period)
        return df
    
    def prepare_market_data(self):
        """기술적 지표 계산"""
        df = self.data.copy()
        
        # 기본 기술적 지표 계산
        df['RSI'] = self.technical.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.technical.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.technical.calculate_bollinger_bands(df['Close'])
        df['Stochastic_K'], df['Stochastic_D'] = self.technical.calculate_stochastic(df)
        df['OBV'] = self.technical.calculate_obv(df)
        
        # 이동평균선
        for period in [5, 10, 20, 60, 120]:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        # 모멘텀 지표
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        return {
            'price': df['Close'].values,
            'volume': df['Volume'].values,
            'indicators': {
                'rsi': df['RSI'].values,
                'macd': df['MACD'].values,
                'macd_signal': df['MACD_Signal'].values,
                'bb_upper': df['BB_Upper'].values,
                'bb_lower': df['BB_Lower'].values,
                'stoch_k': df['Stochastic_K'].values,
                'stoch_d': df['Stochastic_D'].values,
                'obv': df['OBV'].values,
                'momentum': df['Momentum'].values,
                'ma5': df['MA5'].values,
                'ma20': df['MA20'].values,
                'ma60': df['MA60'].values,
                'ma120': df['MA120'].values
            },
            'df': df
        }

class EnhancedInvestmentDecisionMaker:
    def __init__(self):
        self.buy_threshold = 0.6
        self.sell_threshold = 0.4

    def analyze_technical_signals(self, data):
        """기술적 신호 분석"""
        df = data['df']
        latest = df.iloc[-1]
        signals = {}
        
        # RSI 분석
        signals['rsi'] = {
            'value': latest['RSI'],
            'signal': 'buy' if latest['RSI'] < 30 else 'sell' if latest['RSI'] > 70 else 'neutral',
            'weight': 0.15
        }
        
        # MACD 분석
        signals['macd'] = {
            'value': latest['MACD'],
            'signal': 'buy' if latest['MACD'] > latest['MACD_Signal'] else 'sell',
            'weight': 0.15
        }
        
        # 볼린저 밴드 분석
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        signals['bollinger'] = {
            'value': bb_position,
            'signal': 'buy' if bb_position < 0.2 else 'sell' if bb_position > 0.8 else 'neutral',
            'weight': 0.15
        }
        
        # 스토캐스틱 분석
        signals['stochastic'] = {
            'value': latest['Stochastic_K'],
            'signal': 'buy' if latest['Stochastic_K'] < 20 else 'sell' if latest['Stochastic_K'] > 80 else 'neutral',
            'weight': 0.15
        }
        
        # 이동평균선 분석
        ma_signal = 'neutral'
        if latest['MA5'] > latest['MA20'] > latest['MA60']:
            ma_signal = 'buy'
        elif latest['MA5'] < latest['MA20'] < latest['MA60']:
            ma_signal = 'sell'
        
        signals['moving_averages'] = {
            'value': latest['MA20'],
            'signal': ma_signal,
            'weight': 0.2
        }
        
        # 거래량 분석
        volume_ratio = latest['Volume'] / df['Volume'].mean()
        signals['volume'] = {
            'value': volume_ratio,
            'signal': 'buy' if volume_ratio > 1.5 else 'sell' if volume_ratio < 0.5 else 'neutral',
            'weight': 0.2
        }
        
        return signals

    def calculate_probabilities(self, signals):
        """매수/매도/관망 확률 계산"""
        buy_score = 0
        sell_score = 0
        
        for indicator, data in signals.items():
            if data['signal'] == 'buy':
                buy_score += data['weight']
            elif data['signal'] == 'sell':
                sell_score += data['weight']
        
        hold_score = 1 - (buy_score + sell_score)
        
        return {
            'buy': round(buy_score, 3),
            'sell': round(sell_score, 3),
            'hold': round(hold_score, 3)
        }

    def make_decision(self, market_data):
        """최종 투자 결정"""
        signals = self.analyze_technical_signals(market_data)
        probabilities = self.calculate_probabilities(signals)
        
        if probabilities['buy'] > self.buy_threshold:
            decision = 'BUY'
        elif probabilities['sell'] > self.sell_threshold:
            decision = 'SELL'
        else:
            decision = 'HOLD'
            
        return decision, probabilities, signals

@tool("Updated Comprehensive Stock Analysis")
def comprehensive_stock_analysis(ticker: str, period: str) -> str:
    """
    주어진 주식 티커에 대한 업데이트된 종합적인 재무 분석을 수행합니다.
    최신 주가 정보, 재무 지표, 성장률, 밸류에이션 및 주요 비율을 제공합니다.
    가장 최근 영업일 기준의 데이터를 사용합니다.
    
    :param ticker: 분석할 주식의 티커 심볼
    :param period: 데이터 가져오기 기간 (1개월, 3개월, 6개월, 1년)
    :return: 재무 분석 결과를 포함한 문자열
    """
    analyzer = FinancialAnalyzer()
    
    # 주가 데이터 가져오기
    historical_prices, stock, error = analyzer.get_historical_data(ticker, period)
    if error:
        return error
        
    # 최신 주가 정보
    latest_price = historical_prices['Close'].iloc[-1]
    latest_time = historical_prices.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    
    # 재무제표 분석
    financials, error = analyzer.analyze_financials(stock)
    if error:
        return error
    
    # 분석 결과 생성
    analysis_result = f"""
    {ticker} 주식 분석 결과 ({latest_time} 기준)
    
    1. 주가 정보
    - 현재가: ${latest_price:.2f}
    - 52주 최고가: ${historical_prices['High'].max():.2f}
    - 52주 최저가: ${historical_prices['Low'].min():.2f}
    
    2. 주요 재무 지표
    - 매출액: {analyzer.format_number(financials['revenue'])}
    - 영업이익: {analyzer.format_number(financials['operating_income'])}
    - 순이익: {analyzer.format_number(financials['net_income'])}
    - EBITDA: {analyzer.format_number(financials['ebitda'])}
    
    3. 수익성 지표
    - 매출총이익률: {((financials['revenue'] - financials['cost_of_revenue']) / financials['revenue'] * 100):.2f}%
    - 영업이익률: {(financials['operating_income'] / financials['revenue'] * 100):.2f}%
    - 순이익률: {(financials['net_income'] / financials['revenue'] * 100):.2f}%
    """
    
    return analysis_result

current_time = datetime.now()
llm = ChatOpenAI(model="gpt-4o-mini")
invest_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# 재무 분석가
financial_analyst = Agent(
    role="Financial Analyst",
    goal="회사의 재무 상태 및 성과 분석",
    backstory="당신은 재무 제표와 비율을 해석하는 데 전문성을 갖춘 노련한 재무 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[comprehensive_stock_analysis],
    llm=llm,
    max_iter=3,
    allow_delegation=False,
    verbose=True
)

# 시장 분석가
market_analyst = Agent(
    role="Market Analyst",
    goal="회사의 시장 지위 및 업계 동향 분석",
    backstory="당신은 기업/산업 현황 및 경쟁 환경을 전문적으로 분석할 수 있는 숙련된 시장 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[search_tool],
    llm=llm,
    max_iter=3,
    allow_delegation=False,
    verbose=True
)

# 위험 분석가
risk_analyst = Agent(
    role="Risk Analyst",
    goal="주식과 관련된 잠재적 위험 식별 및 평가",
    backstory="당신은 투자에서 명백한 위험과 숨겨진 위험을 모두 식별하는 예리한 안목을 갖춘 신중한 위험 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[comprehensive_stock_analysis],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 투자 전문가
investment_advisor = Agent(
    role="Investment Advisor",
    goal="전체 분석을 기반으로 한 투자 추천 제공",
    backstory="다양한 분석을 종합하여 전략적 투자 조언을 제공하는 신뢰할 수 있는 투자 자문가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    llm=invest_llm,
    allow_delegation=False,
    verbose=True
)

def get_user_input():
    ticker = input("투자 자문을 구하고 싶은 기업명을 입력해주세요: ")
    return ticker

def create_dynamic_tasks(ticker, period):
    financial_analysis = Task(
        description=f"""{ticker}에 대한 철저한 재무 분석을 수행합니다. 
        주요 재무 지표에 집중하세요. 
        회사의 재무 건전성 및 성과 추세에 대한 인사이트를 제공합니다. 날짜: {current_time:%Y년 %m월 %d일}""",
        agent=financial_analyst,
        expected_output=f"""{ticker}의 재무 상태에 대한 종합적인 분석 보고서. 
        주요 재무 지표, 수익성, 부채 비율 등을 포함하며, 
        회사의 재무 건전성과 성과 동향에 대한 인사이트를 제공해야 합니다."""
    )

    market_analysis = Task(
        description=f"""{ticker}의 시장 위치를 분석합니다. 
        경쟁 우위, 시장 점유율, 업계 동향을 평가하세요. 
        회사의 성장 잠재력과 시장 과제에 대한 인사이트를 제공하세요. 날짜: {current_time:%Y년 %m월 %d일}""",
        agent=market_analyst,
        expected_output=f"""{ticker}의 시장 위치에 대한 상세한 분석 보고서. 
        경쟁 우위, 시장 점유율, 산업 동향을 평가하고, 
        회사의 성장 잠재력과 시장 과제에 대한 인사이트를 포함해야 합니다."""
    )

    risk_assessment = Task(
        description=f"""{ticker}에 대한 투자와 관련된 주요 위험을 파악하고 평가합니다. 
        시장 위험, 운영 위험, 재무 위험 및 회사별 위험을 고려하세요. 
        종합적인 위험 프로필을 제공합니다. 날짜: {current_time:%Y년 %m월 %d일}""",
        agent=risk_analyst,
        expected_output=f"""{ticker} 투자와 관련된 주요 리스크에 대한 포괄적인 평가 보고서. 
        시장 리스크, 운영 리스크, 재무 리스크, 회사 특정 리스크를 고려하여 
        종합적인 리스크 분석 결과를 제시해야 합니다."""
    )

    investment_recommendation = Task(
        description=f"""{ticker}의 재무 분석, 시장 분석, 위험 평가를 바탕으로 종합적인 투자 추천을 제공합니다. 
        주식의 잠재 수익률, 위험 및 다양한 유형의 투자자에 대한 적합성을 고려하세요. 한글로 작성하세요. 날짜: {current_time:%Y년 %m월 %d일}""",
        agent=investment_advisor,
        expected_output=f"""
        1. 제목 및 기본 정보
           - 회사명, 티커, 현재 주가, 목표주가, 투자의견 등
        
        2. 요약(Executive Summary)
           - 핵심 투자 포인트와 주요 재무 지표를 간단히 정리
        
        3. 기업 개요
           - 회사의 주요 사업 영역, 연혁, 시장 점유율 등
        
        4. 산업 및 시장 분석
           - 해당 기업이 속한 산업의 트렌드와 전망
        
        5. 재무 분석
           - 매출, 영업이익, 순이익 등 주요 재무지표 분석
           - 수익성, 성장성, 안정성 지표 분석
        
        6. 밸류에이션
           - P/E, P/B, ROE 등 주요 밸류에이션 지표 분석
           - 경쟁사 대비 상대 밸류에이션
        
        7. 투자 의견 및 목표주가
           - 투자의견 제시 및 근거 설명
           - 목표주가 산정 방법과 근거
        
        8. 투자 위험 요인
           - 잠재적인 리스크 요인들을 나열
        
        9. 재무제표 요약
           - 최근 몇 년간의 요약 손익계산서, 재무상태표, 현금흐름표
        """
    )

    return [financial_analysis, market_analysis, risk_assessment, investment_recommendation]

def main():
    st.title("주식 분석 에이전트")
    
    # 사이드바에 LLM 선택
    llm_option = st.sidebar.selectbox("LLM 선택", ["gpt-4o-mini", "claude-3-5-sonnet-20240620"])
    
    # 사이드바에 기업명 입력
    ticker = st.sidebar.text_input("투자 자문을 구하고 싶은 기업명을 입력해주세요:")
    
    # 사이드바에 데이터 가져오기 기간 선택
    period = st.sidebar.selectbox("데이터 가져오기 기간 선택", ["1개월", "3개월", "6개월", "1년"])
    
    # 사이드바에 분석 종류 선택
    analysis_type = st.sidebar.selectbox(
        "분석 종류 선택", 
        ["기술적 분석", "재무 분석", "시장 분석", "위험 분석", "투자 추천"]
    )
    
    if st.sidebar.button("분석 시작"):
        if not ticker:
            st.warning("기업명을 입력해주세요.")
            return
            
        # 데이터 분석기 생성
        analyzer = MarketDataAnalyzer(ticker)
        market_data = analyzer.prepare_market_data()
        
        if analysis_type == "기술적 분석":
            display_technical_analysis(market_data, ticker)
        else:
            # 기존 분석 로직
            decision_maker = EnhancedInvestmentDecisionMaker()
            decision, probabilities, signals = decision_maker.make_decision(market_data)
            display_investment_decision(ticker, market_data, decision, probabilities, signals)

def display_technical_analysis(market_data, ticker):
    """기술적 분석 결과 표시"""
    st.header(f"{ticker} 기술적 분석 결과")
    
    # 현재 가격 정보
    df = market_data['df']
    latest = df.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("가격 정보")
        st.write(f"현재가: ${latest['Close']:.2f}")
        st.write(f"거래량: {latest['Volume']:,.0f}")
        
        st.subheader("이동평균선")
        st.write(f"MA5: ${latest['MA5']:.2f}")
        st.write(f"MA20: ${latest['MA20']:.2f}")
        st.write(f"MA60: ${latest['MA60']:.2f}")
        st.write(f"MA120: ${latest['MA120']:.2f}")
    
    with col2:
        st.subheader("기술적 지표")
        st.write(f"RSI: {latest['RSI']:.2f}")
        st.write(f"MACD: {latest['MACD']:.2f}")
        st.write(f"MACD Signal: {latest['MACD_Signal']:.2f}")
        st.write(f"스토캐스틱 K: {latest['Stochastic_K']:.2f}")
        st.write(f"스토캐스틱 D: {latest['Stochastic_D']:.2f}")
    
    # 볼린저 밴드 정보
    st.subheader("볼린저 밴드")
    st.write(f"상단: ${latest['BB_Upper']:.2f}")
    st.write(f"중간: ${latest['BB_Middle']:.2f}")
    st.write(f"하단: ${latest['BB_Lower']:.2f}")
    
    # 기술적 분석 해석
    st.subheader("기술적 분석 해석")
    
    # RSI 해석
    st.write("RSI 분석:")
    if latest['RSI'] < 30:
        st.write("- 과매도 구간 (매수 고려)")
    elif latest['RSI'] > 70:
        st.write("- 과매수 구간 (매도 고려)")
    else:
        st.write("- 중립 구간")
    
    # MACD 해석
    st.write("MACD 분석:")
    if latest['MACD'] > latest['MACD_Signal']:
        st.write("- 상승 추세 (매수 신호)")
    else:
        st.write("- 하락 추세 (매도 신호)")
    
    # 이동평균선 해석
    st.write("이동평균선 분석:")
    if latest['MA5'] > latest['MA20'] > latest['MA60']:
        st.write("- 상승 추세")
    elif latest['MA5'] < latest['MA20'] < latest['MA60']:
        st.write("- 하락 추세")
    else:
        st.write("- 횡보 추세")

def display_investment_decision(ticker, market_data, decision, probabilities, signals):
    """투자 결정 결과 표시"""
    st.header(f"{ticker} 투자 분석 결과")
    
    # 현재 주가
    st.subheader("현재 시장 정보")
    st.write(f"현재 주가: ${market_data['price'][-1]:.2f}")
    
    # 기술적 지표 분석
    st.subheader("기술적 지표 분석")
    for indicator, data in signals.items():
        st.write(f"{indicator}: {data['signal'].upper()} (값: {data['value']:.2f})")
    
    # 투자 확률
    st.subheader("투자 확률")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("매수 확률", f"{probabilities['buy']:.1%}")
    with col2:
        st.metric("매도 확률", f"{probabilities['sell']:.1%}")
    with col3:
        st.metric("관망 확률", f"{probabilities['hold']:.1%}")
    
    # 최종 투자 결정
    st.subheader("최종 투자 결정")
    st.write(f"추천: {decision}")

if __name__ == "__main__":
    main()
