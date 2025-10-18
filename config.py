import os
from datetime import datetime, timedelta
import json

class Config:
    """프로젝트 전체 설정"""
    
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    PRICE_DATA_DIR = os.path.join(BASE_DIR, 'price_data')
    FEATURE_LOG_DIR = os.path.join(BASE_DIR, 'feature_log')
    TRADE_LOG_DIR = os.path.join(BASE_DIR, 'trade_log')
    RESULT_DIR = os.path.join(BASE_DIR, 'result')
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')    
    # 모델 설정
    PREDICTION_WINDOW = 10  # 10분 예측
    WIN_RATE = 0.80  # 10분 승리시 80% 수익
    TARGET_WIN_RATE = 0.60  # 목표 승률 60%
    RETRAIN_THRESHOLD = 0.55  # 55% 미만시 재학습
    EVALUATION_WINDOW = 50  # 50번 거래마다 평가
    CONF_THRESHOLD = 0.60       # 신뢰도 임계값(60%)
    MAX_CONCURRENT_POS = 5      # 동시 포지션 최대 5
    ENTRY_COOLDOWN_SEC = 60    # 진입 최소 간격(초) - 텀 제어
    ENTRY_CONF_THRESHOLD = 0.60

    # 학습 설정
    INITIAL_TRAIN_MONTHS = 8  # 초기 학습: 8개월
    INITIAL_TEST_MONTHS = 1   # 초기 검증: 1개월
    TRAIN_RATIO = 0.70  # 재학습시 학습 비율
    VAL_RATIO = 0.20    # 재학습시 검증 비율
    TEST_RATIO = 0.10   # 재학습시 테스트 비율
    
    # LightGBM 기본 파라미터
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42,
        'n_estimators': 100,
        'early_stopping_rounds': 10
    }
    
    # 앙상블 설정
    ENSEMBLE_MODELS = 5  # 앙상블에 사용할 모델 수
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.MODEL_DIR, 
            os.path.join(cls.MODEL_DIR, 'backup'),
            cls.PRICE_DATA_DIR,
            os.path.join(cls.PRICE_DATA_DIR, 'raw'),
            os.path.join(cls.PRICE_DATA_DIR, 'processed'),
            cls.FEATURE_LOG_DIR,
            cls.TRADE_LOG_DIR,
            cls.RESULT_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath=None):
        """설정 저장"""
        if filepath is None:
            filepath = os.path.join(cls.BASE_DIR, 'config.json')
        
        config_dict = {
            'prediction_window': cls.PREDICTION_WINDOW,
            'target_win_rate': cls.TARGET_WIN_RATE,
            'retrain_threshold': cls.RETRAIN_THRESHOLD,
            'lgbm_params': cls.LGBM_PARAMS,
            'ensemble_models': cls.ENSEMBLE_MODELS,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)