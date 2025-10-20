"""
바이너리 옵션 예측 시스템 프로젝트 구조

project/
│
├── models/                 # 학습된 모델 저장
│   ├── current_model.pkl   # 현재 사용 중인 모델
│   ├── backup/            # 이전 모델 백업
│   └── metrics.json       # 모델 성능 기록
│
├── price_data/            # 가격 데이터
│   ├── raw/              # 원본 OHLCV 데이터
│   └── processed/        # 전처리된 데이터
│
├── feature_log/          # 피처 로그
│   ├── feature_pool.json # 전체 피처 풀
│   └── selected_features.json # 선택된 피처
│
├── trade_log/            # 거래 기록
│   ├── trades.csv        # 거래 내역 (시간, 예측, 실제, 결과)
│   └── performance.csv   # 성능 추적
│
├── result/               # 최종 데이터셋
│   └── training_data.pkl # 병합된 학습용 데이터
│
├── main_pipe.py          # 메인 파이프라인 (자동화)
├── data_merge.py         # 데이터 병합 모듈
├── model_train.py        # 모델 학습/재학습
├── real_trade.py         # 실시간 거래 및 검증
└── config.py             # 설정 파일
"""

# config.py - 전체 설정 관리
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
    RETRAIN_THRESHOLD = 0.60# 56% 미만시 재학습
    EVALUATION_WINDOW = 50  # 50번 거래마다 평가
    MIN_CONFIDENCE = 0.56 #신뢰도 60이상
    
    # 학습 설정
    INITIAL_TRAIN_MONTHS = 8  # 초기 학습: 8개월
    INITIAL_TEST_MONTHS = 1   # 초기 검증: 1개월
    TRAIN_RATIO = 0.70  # 재학습시 학습 비율
    VAL_RATIO = 0.20    # 재학습시 검증 비율
    TEST_RATIO = 0.10   # 재학습시 테스트 비율

    #연속진입
    TRADE_INTERVAL_MINUTES = 1 # 연손진입 1분제한

    # 바이너리 옵션 페이아웃 (승리 +0.8, 패배 -1 기준)
    PAYOUT_WIN = 0.80
    PAYOUT_LOSS = 1.00
    
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
        'early_stopping_rounds': 10,
        'is_unbalance': True          # ← 추가 (또는 아래 scale_pos_weight 동적 설정)

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

    @classmethod
    def ev_thresholds(cls):
        # EV≥0 기준 임계값 (롱: p_up ≥ 1/(1+win), 숏: p_up ≤ 1 - 그 값)
        th_up = 1.0 / (1.0 + cls.PAYOUT_WIN)
        th_dn = 1.0 - th_up
        return th_up, th_dn