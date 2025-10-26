"""
바이너리 옵션 트레이딩 시스템 - 전역 설정
버전: 1.3.0
"""

import os
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# 시스템 버전 관리
# ============================================================
SYSTEM_VERSION = "1.3.0"
MODEL_VERSION = "1.3.0"
FEATURE_VERSION = "1.3.0"
FILTER_VERSION = "1.3.0"
CUTOFF_VERSION = "1.3.0"
DATA_VERSION = "1.3.0"

# ============================================================
# 디렉토리 구조
# ============================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# 하위 디렉토리
TRADE_LOG_DIR = LOG_DIR / "trade_log"
TRADE_ENTRY_DIR = TRADE_LOG_DIR / "entries"
TRADE_CLOSE_DIR = TRADE_LOG_DIR / "closes"
TRADE_META_DIR = TRADE_LOG_DIR / "meta"
FEATURE_LOG_DIR = LOG_DIR / "feature_log"
MONITOR_LOG_DIR = LOG_DIR / "monitor"
SYSTEM_LOG_DIR = LOG_DIR / "system"

# 자동 생성
DIRS_TO_CREATE = [
    DATA_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR, REPORT_DIR,
    TRADE_LOG_DIR, TRADE_ENTRY_DIR, TRADE_CLOSE_DIR, TRADE_META_DIR,
    FEATURE_LOG_DIR, MONITOR_LOG_DIR, SYSTEM_LOG_DIR
]

for dir_path in DIRS_TO_CREATE:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# 타임프레임 설정
# ============================================================
BAR_MINUTES = 30  # 기준 타임프레임
REGIME_TIMEFRAMES = ['4h', '1h', '15m']  # 멀티 타임프레임 레짐
REGIME_WEIGHTS = {
    '4h': 0.5,   # 장기 추세
    '1h': 0.3,   # 중기 추세
    '15m': 0.2   # 단기 추세
}

# ============================================================
# 레짐 파라미터
# ============================================================
ADX_THRESHOLD = 25.0  # ADX 임계값 (추세 강도)
ADX_STRONG_TREND = 40.0  # 강한 추세
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
REGIME_LOOKBACK = {
    '4h': 4,    # 16시간 (추세 큰 흐름만)
    '1h': 12,   # 12시간 (중기 추세)
    '15m': 24   # 6시간 (단기 변동)
}

# ============================================================
# 데이터 분할 비율
# ============================================================
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 시계열 CV 설정
N_SPLITS = 5
MIN_TRAIN_SIZE = 1000  # 최소 학습 샘플 수

# ============================================================
# 모델 파라미터
# ============================================================
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_child_samples': 20,
    'max_depth': 7,
    'n_estimators': 200
}

# 앙상블 설정
N_ENSEMBLE = 3
CALIBRATION_METHOD = 'isotonic'  # or 'sigmoid'

# ============================================================
# 트레이딩 파라미터
# ============================================================
# 페이아웃
PAYOUT = 0.85  # 30분 이상 승리 시 85%
P_STAR = 1.0 / (1.0 + PAYOUT)  # ≈ 0.5405

DECISION_MARGIN_DEFAULT = 0.02  # p* 기준 마진만 사용


# TTL (Time To Live)
EARLY_EXIT_ENABLED = False
TTL_ENABLED = False
TTL_SECONDS = 0  # 사용 안 함
TTL_EXTENSION_MAX = 0

# 리프랙토리 윈도우
REFRACTORY_WINDOW_SECONDS = 1800  # 5분간 재진입 금지

# 동시 포지션 제한
MAX_CONCURRENT_POSITIONS = 5

TZ = timezone.utc
USE_UTC = True

# ============================================================
# 재학습 트리거
# ============================================================
RETRAIN_CHECK_INTERVAL = 50  # 50거래마다 체크
RETRAIN_WIN_RATE_THRESHOLD = 0.55  # 승률 55% 미만 시 재학습
RETRAIN_CONFIDENCE_LEVEL = 0.95  # 윌슨 하한 신뢰구간
MIN_TRADES_FOR_RETRAIN = 50  # 최소 거래 수

# ============================================================
# 동적 필터
# ============================================================
FILTER_STATE_FILE = CACHE_DIR / "filter_state.json"
LOSS_PATTERN_MIN_COUNT = 10  # 패턴 인식 최소 샘플
LOSS_RATE_THRESHOLD = 0.65  # 손실률 65% 이상 시 필터 활성화

# ============================================================
# 로그 스키마 정의
# ============================================================
TRADE_LOG_COLUMNS = [
    # 식별자
    'trade_id',          # 거래 고유 ID
    'bar30_start',       # 30분봉 시작 시각
    'bar30_end',         # 30분봉 종료 시각
    'entry_ts',          # 진입 타임스탬프
    'label_ts',          # 레이블 확정 타임스탬프
    
    # 인덱스 매핑
    'm1_index_entry',    # 진입 시점 1분봉 인덱스
    'm1_index_label',    # 레이블 시점 1분봉 인덱스
    
    # 가격 정보
    'entry_price',       # 진입 가격
    'label_price',       # 레이블 가격
    'payout',            # 페이아웃 비율
    
    # 결과
    'result',            # WIN / LOSS / OPEN / CANCELLED
    'side',              # UP / DOWN
    
    # 레짐 정보
    'regime',            # 레짐 방향 (1: UP, -1: DOWN, 0: FLAT)
    'is_weekend',        # 주말 여부
    'regime_score',      # ADX 가중 레짐 점수
    'adx',               # ADX 값
    'di_plus',           # DI+
    'di_minus',          # DI-
    
    # 예측 정보
    'p_at_entry',        # 진입 시점 예측 확률
    'refractory_window', # 리프랙토리 윈도우 (초)
    
    # 필터 정보
    'filters_applied',   # 적용된 필터 목록 (JSON)
    'reason_code',       # 진입/청산 사유 코드
    'blocked_reason',    # 차단 사유
    
    # 버전 정보
    'model_ver',         # 모델 버전
    'feature_ver',       # 피처 버전
    'filter_ver',        # 필터 버전
    'cutoff_ver',        # 컷오프 버전
    'data_ver',          # 데이터 버전
    
    # 메타
    'mode',              # LIVE / BACKTEST / PAPER
    'status'             # ACTIVE / CLOSED / CANCELLED
]

FEATURE_LOG_COLUMNS = [
    'bar30_start',       # 30분봉 시작 시각
    'bar30_end',         # 30분봉 종료 시각
    'open', 'high', 'low', 'close', 'volume',  # OHLCV
    
    # 테크니컬 지표
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'stoch_k', 'stoch_d',
    'ema_9', 'ema_21', 'ema_50',
    'sma_20', 'sma_50', 'sma_200',
    
    # 레짐 정보
    'regime_4h', 'regime_1h', 'regime_15m',
    'regime_final', 'regime_score',
    'adx', 'di_plus', 'di_minus',
    
    # 타겟
    'target',            # 0: DOWN, 1: UP
    
    # 버전
    'feature_ver',
    'data_ver'
]

# ============================================================
# 타입 스키마 (dtype 정규화)
# ============================================================
PRICE_DATA_DTYPES = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64'
}

TRADE_LOG_DTYPES = {
    'trade_id': 'str',
    'bar30_start': 'datetime64[ns]',
    'bar30_end': 'datetime64[ns]',
    'entry_ts': 'datetime64[ns]',
    'label_ts': 'datetime64[ns]',
    'm1_index_entry': 'int64',
    'm1_index_label': 'int64',
    'entry_price': 'float64',
    'label_price': 'float64',
    'payout': 'float64',
    'result': 'str',
    'side': 'str',
    'regime': 'int64',
    'is_weekend': 'bool',
    'regime_score': 'float64',
    'adx': 'float64',
    'di_plus': 'float64',
    'di_minus': 'float64',
    'p_at_entry': 'float64',
    'refractory_window': 'int64',
    'filters_applied': 'str',
    'reason_code': 'str',
    'blocked_reason': 'str',
    'model_ver': 'str',
    'feature_ver': 'str',
    'filter_ver': 'str',
    'cutoff_ver': 'str',
    'data_ver': 'str',
    'mode': 'str',
    'status': 'str'
}


REASON_CODES = {
    # 진입
    'ENTRY_HIGH_PROB': '고확률 진입',
    'ENTRY_REGIME_UP': '상승 레짐 진입',
    'ENTRY_REGIME_DOWN': '하락 레짐 진입',
    'ENTRY_REGIME_FLET': '횡보 레짐 진입',
    
    # 차단
    'BLOCKED_SPREAD': '스프레드 과다',
    'BLOCKED_SHOCK': '급등락',
    'BLOCKED_VOLATILITY': '변동성 과다',
    'BLOCKED_REFRACTORY': '리프랙토리 기간',
    'BLOCKED_LOW_PROB': '확률 부족',
    
    # 정산
    'CLOSE_EXPIRY_WIN': '만기 승리',
    'CLOSE_EXPIRY_LOSS': '만기 패배',
    
    # 예외
    'CANCELLED_ERROR': '시스템 오류',
}

# ============================================================
# 모니터링 설정
# ============================================================
MONITOR_UPDATE_INTERVAL = 300  # 5분마다 업데이트
MONITOR_SNAPSHOT_INTERVAL = 3600  # 1시간마다 스냅샷
REPORT_INTERVAL = 14400  # 4시간마다 리포트 생성

# 경고 임계값
ALERT_WIN_RATE_LOW = 0.45
ALERT_CONSECUTIVE_LOSSES = 5
ALERT_NO_ENTRY_HOURS = 2
ALERT_REGIME_BIAS = 0.8
ALERT_NAN_RATIO = 0.05

# ============================================================
# API 설정 (환경 변수로 관리 권장)
# ============================================================
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

# ============================================================
# 유틸리티 함수
# ============================================================
def get_config_summary():
    """설정 요약 정보 반환"""
    return {
        'system_version': SYSTEM_VERSION,
        'model_version': MODEL_VERSION,
        'feature_version': FEATURE_VERSION,
        'bar_minutes': BAR_MINUTES,
        'regime_timeframes': REGIME_TIMEFRAMES,
        'max_positions': MAX_CONCURRENT_POSITIONS,
        'ttl_seconds': TTL_SECONDS,
        'retrain_threshold': RETRAIN_WIN_RATE_THRESHOLD
    }

def validate_config():
    """설정 유효성 검증"""
    errors = []
    
    # 버전 형식 체크
    for ver_name, ver_value in [
        ('SYSTEM_VERSION', SYSTEM_VERSION),
        ('MODEL_VERSION', MODEL_VERSION),
        ('FEATURE_VERSION', FEATURE_VERSION)
    ]:
        if not isinstance(ver_value, str) or len(ver_value.split('.')) != 3:
            errors.append(f"{ver_name} 형식 오류: {ver_value}")
    
    # 비율 합계 체크
    if abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) > 0.001:
        errors.append("데이터 분할 비율 합계가 1.0이 아님")
    
    # 레짐 가중치 합계 체크
    if abs(sum(REGIME_WEIGHTS.values()) - 1.0) > 0.001:
        errors.append("레짐 가중치 합계가 1.0이 아님")
    
    # 디렉토리 존재 확인
    for dir_path in DIRS_TO_CREATE:
        if not dir_path.exists():
            errors.append(f"디렉토리 생성 실패: {dir_path}")
    
    # 로그 스키마 길이 체크
    if len(TRADE_LOG_COLUMNS) != len(set(TRADE_LOG_COLUMNS)):
        errors.append("TRADE_LOG_COLUMNS에 중복 항목 존재")
    
    if len(FEATURE_LOG_COLUMNS) != len(set(FEATURE_LOG_COLUMNS)):
        errors.append("FEATURE_LOG_COLUMNS에 중복 항목 존재")
    
    return len(errors) == 0, errors

# ============================================================
# 초기화 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("바이너리 옵션 트레이딩 시스템 설정 검증")
    print("=" * 60)
    
    valid, errors = validate_config()
    
    if valid:
        print("✅ 설정 검증 통과")
        print("\n📋 설정 요약:")
        for key, value in get_config_summary().items():
            print(f"  - {key}: {value}")
    else:
        print("❌ 설정 검증 실패:")
        for error in errors:
            print(f"  - {error}")
    
    print("\n📁 디렉토리 구조:")
    for dir_path in DIRS_TO_CREATE:
        status = "✅" if dir_path.exists() else "❌"
        print(f"  {status} {dir_path}")
    
    print("\n📊 로그 스키마:")
    print(f"  - TRADE_LOG_COLUMNS: {len(TRADE_LOG_COLUMNS)}개")
    print(f"  - FEATURE_LOG_COLUMNS: {len(FEATURE_LOG_COLUMNS)}개")

    
def dynamic_margin(ece50: float = 0.0, entropy: float = 0.0) -> float:
    import numpy as np
    base = 0.02
    return float(np.clip(base + 0.5*ece50 + 0.02*entropy, 0.01, 0.03))

def decide(p_cal: float, margin: float) -> str | None:
    if p_cal >= P_STAR + margin: 
        return "UP"
    if p_cal <= (1.0 - P_STAR) - margin: 
        return "DOWN"
    return None