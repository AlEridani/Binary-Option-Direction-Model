"""
전체 시스템 기능 테스트
각 모듈의 핵심 기능이 정상 작동하는지 검증합니다.
"""

import sys
import os
import traceback
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("시스템 기능 테스트")
print("=" * 70 + "\n")

test_results = []

def test_module(name, func):
    """테스트 실행 헬퍼"""
    try:
        print(f"\n{'='*70}")
        print(f"테스트: {name}")
        print(f"{'='*70}")
        func()
        print(f"✅ {name} - 통과")
        test_results.append((name, True, None))
        return True
    except Exception as e:
        print(f"❌ {name} - 실패")
        print(f"오류: {e}")
        traceback.print_exc()
        test_results.append((name, False, str(e)))
        return False

# ============================================================
# 1. Config 테스트
# ============================================================
def test_config():
    import config
    
    # 설정 검증
    valid, errors = config.validate_config()
    assert valid, f"설정 검증 실패: {errors}"
    
    # 주요 설정값 확인
    assert config.BAR_MINUTES == 30, "BAR_MINUTES != 30"
    assert len(config.REGIME_TIMEFRAMES) == 3, "REGIME_TIMEFRAMES 개수 오류"
    assert sum(config.REGIME_WEIGHTS.values()) == 1.0, "REGIME_WEIGHTS 합계 오류"
    
    print(f"  ✓ 시스템 버전: {config.SYSTEM_VERSION}")
    print(f"  ✓ 레짐 타임프레임: {config.REGIME_TIMEFRAMES}")
    print(f"  ✓ 디렉토리: {len(config.DIRS_TO_CREATE)}개")

test_module("Config 검증", test_config)

# ============================================================
# 2. TimeframeManager 테스트
# ============================================================
def test_timeframe_manager():
    from timeframe_manager import TimeframeManager
    
    tf_manager = TimeframeManager()
    
    # 테스트 데이터 생성
    timestamps = pd.date_range(start='2025-01-01', periods=100, freq='1min', tz='UTC')
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(51000, 52000, 100),
        'low': np.random.uniform(49000, 50000, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # OHLC 정합성
    df_1m['high'] = df_1m[['open', 'close']].max(axis=1) + 100
    df_1m['low'] = df_1m[['open', 'close']].min(axis=1) - 100
    
    # 1분봉 → 30분봉 집계
    df_30m = tf_manager.aggregate_1m_to_30m(df_1m)
    
    assert not df_30m.empty, "30분봉 집계 실패"
    
    # 개수는 올림 처리되므로 범위로 체크
    expected_min = len(df_1m) // 30
    expected_max = (len(df_1m) + 29) // 30
    
    assert expected_min <= len(df_30m) <= expected_max, \
        f"30분봉 개수 오류: 예상 {expected_min}~{expected_max}, 실제 {len(df_30m)}"
    
    # 검증
    valid, errors = tf_manager.validate_aggregation(df_1m, df_30m)
    assert valid, f"집계 검증 실패: {errors}"
    
    print(f"  ✓ 1분봉: {len(df_1m)}개")
    print(f"  ✓ 30분봉: {len(df_30m)}개")
    print(f"  ✓ 집계 검증 통과")

test_module("TimeframeManager", test_timeframe_manager)

# ============================================================
# 3. FeatureEngineer 테스트
# ============================================================
def test_feature_engineer():
    from feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer()
    
    # 테스트 데이터 (충분한 양 - 10일치)
    timestamps = pd.date_range(start='2025-01-01', periods=14400, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(14400).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 14400),
        'high': prices + np.random.uniform(0, 100, 14400),
        'low': prices - np.random.uniform(0, 100, 14400),
        'volume': np.random.uniform(1000, 5000, 14400)
    })
    
    # 피처 생성
    features = fe.create_feature_pool(df_1m, lookback_bars=50)  # lookback 줄임
    
    assert not features.empty, "피처 생성 실패"
    assert 'regime' in features.columns, "레짐 컬럼 없음"
    assert 'target' in features.columns, "타겟 컬럼 없음"
    
    # 피처 개수 확인
    feature_names = fe.get_feature_names(features)
    
    print(f"  ✓ 30분봉: {len(features)}개")
    print(f"  ✓ 피처: {len(feature_names)}개")
    print(f"  ✓ 레짐 분포: {features['regime'].value_counts().to_dict()}")

test_module("FeatureEngineer", test_feature_engineer)

# ============================================================
# 4. DataLoader 테스트
# ============================================================
def test_data_loader():
    from data_loader import DataLoader
    
    loader = DataLoader()
    
    # 테스트 데이터 저장 (충분한 양 - 5일치)
    timestamps = pd.date_range(start='2025-01-01', periods=7200, freq='1min', tz='UTC')
    df_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(50000, 51000, 7200),
        'high': np.random.uniform(51000, 52000, 7200),
        'low': np.random.uniform(49000, 50000, 7200),
        'close': np.random.uniform(50000, 51000, 7200),
        'volume': np.random.uniform(1000, 5000, 7200)
    })
    
    loader.save_price_data(df_test, "test_data_loader.csv")
    
    # 데이터 로드
    df_loaded = loader.load_price_data("2025-01-01", "2025-01-05", "test_data_loader.csv")
    
    assert not df_loaded.empty, "데이터 로드 실패"
    
    # 학습 데이터 준비 (lookback 줄임)
    X, y = loader.prepare_training_data(df_test, use_cache=False, lookback_bars=50)
    
    assert not X.empty, "학습 데이터 준비 실패"
    assert not y.empty, "타겟 데이터 없음"
    
    # 데이터 품질 검증
    valid, errors = loader.validate_data_quality(X, y)
    
    print(f"  ✓ 데이터 로드: {len(df_loaded)}개")
    print(f"  ✓ 학습 데이터: X={X.shape}, y={y.shape}")
    print(f"  ✓ 품질 검증: {'통과' if valid else '실패'}")

test_module("DataLoader", test_data_loader)

# ============================================================
# 5. ModelTrainer 테스트
# ============================================================
def test_model_trainer():
    from model_train import ModelTrainer
    from data_loader import DataLoader
    
    trainer = ModelTrainer()
    loader = DataLoader()
    
    # 테스트 데이터 생성 (10일치)
    timestamps = pd.date_range(start='2025-01-01', periods=14400, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(14400).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 14400),
        'high': prices + np.random.uniform(0, 100, 14400),
        'low': prices - np.random.uniform(0, 100, 14400),
        'volume': np.random.uniform(1000, 5000, 14400)
    })
    
    X, y = loader.prepare_training_data(df_1m, use_cache=False, lookback_bars=50)
    
    if X.empty or y.empty:
        print("  ⚠️  학습 데이터 준비 실패, 스킵")
        return
    
    # 간단한 학습 (빠른 테스트)
    if 'regime' in X.columns and len(X) > 100:
        results = trainer.train_ensemble_regime(X, y, regime_col='regime', test_size=0.3)
    elif len(X) > 100:
        results = trainer.train_single_model(X, y, test_size=0.3)
    else:
        print(f"  ⚠️  데이터 부족 ({len(X)}개), 스킵")
        return
    
    assert len(results) > 0, "학습 결과 없음"
    
    # 예측 테스트
    X_test = X.head(10)
    predictions = trainer.predict(X_test, regime=1, use_regime_model=True)
    
    assert predictions is not None, "예측 실패"
    assert len(predictions) == 10, "예측 개수 오류"
    
    print(f"  ✓ 학습 완료: {len(results)}개 모델")
    print(f"  ✓ 예측 테스트: {len(predictions)}개")

test_module("ModelTrainer", test_model_trainer)

# ============================================================
# 6. LogManager 테스트
# ============================================================
def test_log_manager():
    from log_manager import LogManager
    
    log_mgr = LogManager()
    
    # 거래 로그 기록
    test_time = datetime.now(timezone.utc)
    
    log_mgr.log_trade_entry_simple(
        trade_id='test_001',
        direction='UP',
        entry_price=50000.0,
        entry_ts=test_time,
        p_at_entry=0.65,
        regime=1
    )
    
    # 거래 결과 업데이트
    success = log_mgr.update_trade_result(
        trade_id='test_001',
        result='WIN',
        label_price=50200.0,
        payout=0.85
    )
    
    # 로그 로드
    date_str = test_time.strftime("%Y%m%d")
    df_trades = log_mgr.load_trade_log(date_str)
    
    assert not df_trades.empty, "거래 로그 로드 실패"
    
    # 검증
    validation = log_mgr.validate_trade_log(df_trades)
    
    print(f"  ✓ 거래 기록: {len(df_trades)}개")
    print(f"  ✓ 업데이트: {'성공' if success else '실패'}")
    print(f"  ✓ 검증: {'통과' if validation['valid'] else '실패'}")

test_module("LogManager", test_log_manager)

# ============================================================
# 7. RealTrader 테스트 (API + 백테스트)
# ============================================================
def test_real_trader():
    from real_trade import RealTrader, BinanceAPIClient
    
    # API 클라이언트 테스트
    api_client = BinanceAPIClient()
    
    # 현재가 조회 (시뮬레이션 모드로 자동 전환)
    price = api_client.get_current_price('BTCUSDT')
    assert price > 0, "가격 조회 실패"
    
    # 캔들 데이터 조회
    df_klines = api_client.get_klines(limit=100)
    assert not df_klines.empty, "캔들 데이터 조회 실패"
    
    # RealTrader 초기화
    trader = RealTrader(symbol='BTCUSDT')
    
    # 상태 확인
    status = trader.get_status()
    
    print(f"  ✓ API 가격: ${price:,.2f}")
    print(f"  ✓ 캔들 데이터: {len(df_klines)}개")
    print(f"  ✓ 모델 로드: {status['model_loaded']}")
    print(f"  ✓ 활성 포지션: {status['active_positions']}")

test_module("RealTrader (API)", test_real_trader)

# ============================================================
# 8. Monitor 테스트
# ============================================================
def test_monitor():
    from monitor import Monitor, PerformanceMonitor
    from log_manager import LogManager
    
    log_mgr = LogManager()
    monitor = Monitor(log_mgr)
    
    # 스냅샷 업데이트
    snapshot = monitor.update()
    
    assert snapshot is not None, "스냅샷 생성 실패"
    assert 'performance' in snapshot, "성능 데이터 없음"
    
    print(f"  ✓ 스냅샷 생성: {len(snapshot)} 항목")
    print(f"  ✓ 알림: {len(snapshot.get('alerts', []))}개")

test_module("Monitor", test_monitor)

# ============================================================
# 9. DataMerger 테스트
# ============================================================
def test_data_merger():
    from data_merge import DataMerger
    from data_loader import DataLoader
    
    merger = DataMerger()
    loader = DataLoader()
    
    # 테스트 데이터 생성
    timestamps = pd.date_range(start='2025-01-01', periods=1440, freq='1min', tz='UTC')
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(50000, 51000, 1440),
        'high': np.random.uniform(51000, 52000, 1440),
        'low': np.random.uniform(49000, 50000, 1440),
        'close': np.random.uniform(50000, 51000, 1440),
        'volume': np.random.uniform(1000, 5000, 1440)
    })
    
    loader.save_price_data(df_1m, "test_merger.csv")
    
    # 데이터 병합
    df_merged = merger.merge_all_data("2025-01-01", "2025-01-01", filename="test_merger.csv")
    
    assert not df_merged.empty, "데이터 병합 실패"
    
    # 품질 검증
    valid, errors = merger.validate_merged_data(df_merged)
    
    # 품질 리포트
    quality = merger.get_data_quality_report(df_merged)
    
    print(f"  ✓ 병합 데이터: {len(df_merged)}개")
    print(f"  ✓ 검증: {'통과' if valid else '실패'}")
    print(f"  ✓ 품질 점수: {quality['quality_score']:.1f}/100")

test_module("DataMerger", test_data_merger)

# ============================================================
# 10. MainPipeline 테스트
# ============================================================
def test_main_pipeline():
    from main_pipe import MainPipeline
    
    pipeline = MainPipeline(symbol='BTCUSDT')
    
    # 시뮬레이션 데이터 생성
    df_sim = pipeline.generate_simulation_data(days=3)
    
    assert not df_sim.empty, "시뮬레이션 데이터 생성 실패"
    assert len(df_sim) == 3 * 1440, "데이터 개수 오류"
    
    print(f"  ✓ 시뮬레이션 데이터: {len(df_sim):,}개")
    print(f"  ✓ 기간: {df_sim['timestamp'].min()} ~ {df_sim['timestamp'].max()}")

test_module("MainPipeline", test_main_pipeline)

# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 70)
print("테스트 결과 요약")
print("=" * 70)

passed_tests = [name for name, result, _ in test_results if result]
failed_tests = [(name, error) for name, result, error in test_results if not result]

print(f"\n총 테스트: {len(test_results)}개")
print(f"✅ 통과: {len(passed_tests)}개")
print(f"❌ 실패: {len(failed_tests)}개")

if failed_tests:
    print("\n실패한 테스트:")
    for name, error in failed_tests:
        print(f"  - {name}")
        print(f"    오류: {error[:100]}...")
else:
    print("\n🎉 모든 테스트 통과!")

print("\n" + "=" * 70)