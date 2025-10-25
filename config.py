"""
config.py - 30분봉 바이너리 옵션 시스템 설정
기존 시스템 + 30분봉 전환 통합
"""

import os
from datetime import datetime, timezone
import json
from pathlib import Path


class Config:
    """프로젝트 전체 설정"""
    
    # ==========================================
    # 기본 경로 설정
    # ==========================================
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / 'models'
    PRICE_DATA_DIR = BASE_DIR / 'price_data'
    FEATURE_LOG_DIR = BASE_DIR / 'feature_log'
    TRADE_LOG_DIR = BASE_DIR / 'trade_log'
    RESULT_DIR = BASE_DIR / 'result'
    BACKUP_DIR = MODEL_DIR / 'backup'
    VERSION_DIR = MODEL_DIR / 'versions'
    
    # API 설정
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')
    
    # ==========================================
    # 타임프레임 설정 (30분 전환)
    # ==========================================
    TIMEFRAME = '30min'  # 메인 타임프레임 (10분 → 30분)
    BASE_TIMEFRAME = '1min'  # 원본 데이터
    PREDICTION_WINDOW = 30  # 예측 윈도우 (30분)
    BAR_MINUTES = 30  # 1 bar = 30분
    
    # ==========================================
    # 타임라인 (t+1 바 예측)
    # ==========================================
    # t: 현재 30분봉
    # 예측 시점: t의 close (bar30_end)
    # 진입: t+1의 open (bar30_end)
    # 만기: t+1의 close (bar30_end + 30min)
    
    HORIZON_MINUTES = 30  # 라벨 호라이즌
    
    # ==========================================
    # 히스테리시스 + TTL 설정
    # ==========================================
    CUT_ON = 0.6   # 진입 임계값
    CUT_OFF = 0.58  # 해제 임계값
    
    # TTL (Time To Live)
    TTL_MIN_SECONDS = 360  # 최소 6분
    TTL_MAX_SECONDS = 540  # 최대 9분
    
    # Δp 최소값
    DP_MIN = 0.01  # p_now - p_prev >= 0.01
    
    # Refractory Period
    REFRACTORY_MINUTES = 30  # 재진입 금지 시간
    
    # ==========================================
    # 업데이트 주기
    # ==========================================
    UPDATE_INTERVAL_SECONDS = 60  # 1분마다 체크
    TRADE_INTERVAL_MINUTES = 1  # 기존 호환
    
    # ==========================================
    # 레짐 설정 (멀티타임프레임)
    # ==========================================
    REGIME_TIMEFRAMES = ['4h', '1h', '30min', '15min']
    REGIME_WEIGHTS = {
        '4h': 0.4,
        '1h': 0.3,
        '30min': 0.2,
        '15min': 0.1
    }
    REGIME_ADX_THR = 20
    REGIME_ADX_WINDOW = 14
    #레짐 임계값
    REGIME_SCORE_UP = 0.30
    REGIME_SCORE_DOWN = -0.30
    
    # ==========================================
    # 바이너리 옵션 설정
    # ==========================================
    WIN_RATE = 0.80  # 승리 시 80% 수익
    PAYOUT_WIN = 0.80
    PAYOUT_LOSS = 1.00
    PAYOUT_RATIO = 0.85  # 페이아웃 비율
    
    # 손익분기점
    BREAKEVEN_WIN_RATE = 1 / (1 + PAYOUT_RATIO)  # ~0.54
    TARGET_WIN_RATE = 0.60  # 목표 승률
    MIN_CONFIDENCE = 0.56  # 최소 신뢰도
    
    # ==========================================
    # 재학습 설정 (50번 거래 기반)
    # ==========================================
    RETRAIN_CHECK_INTERVAL = 50  # 50번 거래마다 체크
    RETRAIN_MIN_WIN_RATE = 0.56  # 승률 56% 미만 시 재학습
    RETRAIN_THRESHOLD = 0.60  # 기존 호환
    EVALUATION_WINDOW = 50  # 평가 윈도우
    
    # 레짐별 최소 샘플
    RETRAIN_MIN_SAMPLES_PER_REGIME = 200
    
    # 재학습 전 자동 검사
    RETRAIN_PSI_THRESHOLD = 0.2
    RETRAIN_KS_THRESHOLD = 0.01
    
    # 재학습 후 회귀 테스트
    RETRAIN_ENTRY_RATE_TOLERANCE = 0.10
    
    # ==========================================
    # 학습 설정
    # ==========================================
    INITIAL_TRAIN_MONTHS = 8
    INITIAL_TEST_MONTHS = 1
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.20
    TEST_RATIO = 0.10
    
    # ==========================================
    # 모델 설정
    # ==========================================
    ENSEMBLE_MODELS = 3  # 앙상블 모델 개수
    TOP_K_FEATURES = 30  # 선택할 피처 개수
    
    # 캘리브레이션
    CALIBRATION_METHOD = 'isotonic'  # 'isotonic' or 'platt'
    
    # LightGBM 파라미터
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
        'is_unbalance': True
    }
    
    # ==========================================
    # 버전 관리
    # ==========================================
    VERSION_FORMAT = "%Y%m%d_%H%M%S"
    
    VERSION_ITEMS = [
        'model_ver',
        'feature_ver',
        'filter_ver',
        'cutoff_ver',
        'data_ver'
    ]
    
    # ==========================================
    # 버그 조기탐지
    # ==========================================
    REPLAY_ENABLED = True
    REPLAY_SCHEDULE_HOUR = 3
    
    SHADOW_MODE = False
    
    CANARY_ENABLED = True
    CANARY_RATIO = 0.2
    CANARY_DURATION_HOURS = 48
    
    ALERT_ENTRY_RATE_CHANGE = 0.3
    ALERT_REGIME_BIAS = 0.8
    ALERT_NAN_TOLERANCE = 0
    
    # ==========================================
    # 로그 설정
    # ==========================================
    LOG_SPLIT_BY_DATE = True
    LOG_TIMEZONE = timezone.utc
    
    # 트레이드 로그 컬럼
    TRADE_LOG_COLUMNS = [
        'trade_id',
        'bar30_start',
        'bar30_end',
        'entry_ts',
        'label_ts',
        'm1_index_entry',
        'm1_index_label',
        'entry_price',
        'label_price',
        'payout',
        'result',
        'side', 
        'regime',
        'is_weekend',
        'regime_score',
        'adx',
        'di_plus',
        'di_minus',
        'p_at_entry',
        'dp_at_entry',
        'cut_on',
        'cut_off',
        'cross_time',
        'ttl_used_sec',
        'ttl_valid',
        'refractory_window',
        'filters_applied',
        'reason_code',
        'blocked_reason',
        'model_ver',
        'feature_ver',
        'filter_ver',
        'cutoff_ver',
        'data_ver',
        'mode',
        'status'         
    ]

    # --- 로그 스키마: 피처 ---
    FEATURE_LOG_COLUMNS = [
        'bar30_start',
        'bar30_end',
        'pred_ts',
        'entry_ts',
        'label_ts',
        'm1_index_entry',
        'm1_index_label',
        'cut_on',
        'cut_off',
        'p_prev',
        'p_now',
        'p_cal',
        'dp',
        'dmin',
        'regime',
        'vol_ratio',
        'spread_bps',
        'vwap_gap_bps',
        'filters_passed',
        'signal_id',
        'model_ver',
        'feature_ver',
        'filter_ver',
        'cutoff_ver',
        'data_ver'
    ]
    
    # ==========================================
    # 유틸리티 메서드
    # ==========================================
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.MODEL_DIR,
            cls.MODEL_DIR / 'backup',
            cls.PRICE_DATA_DIR,
            cls.PRICE_DATA_DIR / 'raw',
            cls.PRICE_DATA_DIR / 'processed',
            cls.FEATURE_LOG_DIR,
            cls.TRADE_LOG_DIR,
            cls.RESULT_DIR,
            cls.BACKUP_DIR,
            cls.VERSION_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"✓ 디렉토리 생성 완료: {cls.BASE_DIR}")
    
    @classmethod
    def get_log_path(cls, log_type: str, date_str: str = None) -> Path:
        """날짜별 로그 파일 경로 반환"""
        if date_str is None:
            date_str = datetime.now(cls.LOG_TIMEZONE).strftime("%Y%m%d")
        
        if log_type == 'trade':
            return cls.TRADE_LOG_DIR / f"{date_str}_trades.csv"
        elif log_type == 'feature':
            return cls.FEATURE_LOG_DIR / f"{date_str}_features.csv"
        else:
            raise ValueError(f"Unknown log type: {log_type}")
    
    @classmethod
    def get_version_string(cls) -> str:
        """현재 버전 문자열 생성"""
        return datetime.now(cls.LOG_TIMEZONE).strftime(cls.VERSION_FORMAT)
    
    @classmethod
    def validate_config(cls):
        """설정 검증"""
        errors = []
        
        if cls.CUT_ON <= cls.CUT_OFF:
            errors.append(f"CUT_ON ({cls.CUT_ON}) must be > CUT_OFF ({cls.CUT_OFF})")
        
        if cls.TTL_MIN_SECONDS >= cls.TTL_MAX_SECONDS:
            errors.append(f"TTL_MIN_SECONDS must be < TTL_MAX_SECONDS")
        
        if cls.BAR_MINUTES != 30:
            errors.append(f"BAR_MINUTES must be 30 for 30min system")
        
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))
        
        print("✓ 설정 검증 완료")
    
    @classmethod
    def print_config(cls):
        """주요 설정 출력"""
        print("\n" + "="*60)
        print("30분봉 바이너리 옵션 트레이딩 시스템 설정")
        print("="*60)
        print(f"타임프레임: {cls.TIMEFRAME} (예측: {cls.PREDICTION_WINDOW}분)")
        print(f"히스테리시스: 진입 {cls.CUT_ON} / 해제 {cls.CUT_OFF}")
        print(f"TTL: {cls.TTL_MIN_SECONDS}~{cls.TTL_MAX_SECONDS}초")
        print(f"재진입 금지: {cls.REFRACTORY_MINUTES}분")
        print(f"레짐: {cls.REGIME_TIMEFRAMES} (가중치: {cls.REGIME_WEIGHTS})")
        print(f"재학습: {cls.RETRAIN_CHECK_INTERVAL}번 거래 후 승률 < {cls.RETRAIN_MIN_WIN_RATE:.0%}")
        print(f"목표 승률: {cls.TARGET_WIN_RATE:.0%}")
        print("="*60 + "\n")
    
    @classmethod
    def save_config(cls, filepath=None):
        """설정 저장"""
        if filepath is None:
            filepath = cls.BASE_DIR / 'config.json'
        
        config_dict = {
            'timeframe': cls.TIMEFRAME,
            'prediction_window': cls.PREDICTION_WINDOW,
            'cut_on': cls.CUT_ON,
            'cut_off': cls.CUT_OFF,
            'target_win_rate': cls.TARGET_WIN_RATE,
            'retrain_check_interval': cls.RETRAIN_CHECK_INTERVAL,
            'retrain_min_win_rate': cls.RETRAIN_MIN_WIN_RATE,
            'lgbm_params': cls.LGBM_PARAMS,
            'ensemble_models': cls.ENSEMBLE_MODELS,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ 설정 저장: {filepath}")
    
    @classmethod
    def ev_thresholds(cls):
        """EV 기준 임계값 (기존 호환)"""
        th_up = 1.0 / (1.0 + cls.PAYOUT_WIN)
        th_dn = 1.0 - th_up
        return th_up, th_dn


# ==========================================
# 초기화
# ==========================================
if __name__ == "__main__":
    Config.create_directories()
    Config.validate_config()
    Config.print_config()
    Config.save_config()