import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Chrome Driver Settings
CHROME_OPTIONS = {
    "headless": True,
    "window_size": (375, 812),
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1",
    "disable_gpu": True,
    "no_sandbox": True
}

# File Paths
DATA_FOLDER = '날짜별 데이터'
STOCK_DATA_FILE = 'stock_data.xlsx'
