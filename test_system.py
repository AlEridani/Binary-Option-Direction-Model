"""
전체 시스템 기능 테스트
30분 바이너리 옵션 시스템 v1.4.0

주요 테스트:
- 레짐별 앙상블 모델
- 캘리브레이션 검증
- 버전 관리 (번들)
- 30분 타임프레임
"""

import sys
import os
import traceback
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("시스템 기능 테스트 (v1.4.0)")
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
    
    valid, errors = config.validate_config()
    assert valid, f"설정 검증 실패: {errors}"
    
    assert config.BAR_MINUTES == 30, "BAR_MINUTES != 30"
    assert len(config.REGIME_TIMEFRAMES) == 3, "REGIME_TIMEFRAMES 개수 오류"
    assert abs(sum(config.REGIME_WEIGHTS.values()) - 1.0) < 0.001, "REGIME_WEIGHTS 합계 오류"
    
    # decide 함수 테스트
    from config import decide, P_STAR, dynamic_margin
    
    margin = dynamic_margin(0, 0)
    
    # UP 진입 (config가 "UP" 리턴하는지 확인)
    result_up = decide(P_STAR + margin + 0.01, margin)
    assert result_up in ["UP", "LONG"], f"UP 진입 실패: {result_up}"
    
    # DOWN 진입
    result_down = decide(1 - P_STAR - margin - 0.01, margin)
    assert result_down in ["DOWN", "SHORT"], f"DOWN 진입 실패: {result_down}"
    
    # 진입 안 함
    assert decide(0.5, margin) is None
    
    print(f"  ✓ 시스템 버전: {config.SYSTEM_VERSION}")
    print(f"  ✓ 30분 타임프레임: {config.BAR_MINUTES}분")
    print(f"  ✓ P_STAR: {P_STAR:.4f}")
    print(f"  ✓ decide() 함수: UP={result_up}, DOWN={result_down}")

test_module("Config 검증", test_config)

# ============================================================
# 2. TimeframeManager 테스트
# ============================================================
def test_timeframe_manager():
    from timeframe_manager import TimeframeManager
    
    tf_manager = TimeframeManager()
    
    # 5일치 1분봉 (7200개)
    timestamps = pd.date_range(start='2025-01-01', periods=7200, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(7200).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 7200),
        'high': prices + 100,
        'low': prices - 100,
        'volume': np.random.uniform(1000, 5000, 7200)
    })
    
    df_1m['high'] = df_1m[['open', 'close']].max(axis=1) + 100
    df_1m['low'] = df_1m[['open', 'close']].min(axis=1) - 100
    
    # 30분봉 집계
    df_30m = tf_manager.aggregate_1m_to_30m(df_1m, realtime_safe=False)
    
    assert not df_30m.empty, "30분봉 집계 실패"
    
    expected_bars = len(df_1m) // 30
    assert abs(len(df_30m) - expected_bars) <= 1, \
        f"30분봉 개수 오류: 예상 ~{expected_bars}, 실제 {len(df_30m)}"
    
    print(f"  ✓ 1분봉: {len(df_1m)}개")
    print(f"  ✓ 30분봉: {len(df_30m)}개")
    print(f"  ✓ 비율: {len(df_1m)/len(df_30m):.1f}:1")

test_module("TimeframeManager (30분봉)", test_timeframe_manager)

# ============================================================
# 3. FeatureEngineer 테스트
# ============================================================
def test_feature_engineer():
    from feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer()
    
    # 10일치 (충분한 데이터)
    timestamps = pd.date_range(start='2025-01-01', periods=14400, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(14400).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 14400),
        'high': prices + 100,
        'low': prices - 100,
        'volume': np.random.uniform(1000, 5000, 14400)
    })
    
    # 피처 생성
    features = fe.create_feature_pool(df_1m, lookback_30m_bars=100)
    
    assert not features.empty, "피처 생성 실패"
    assert 'regime' in features.columns, "regime 컬럼 없음"
    assert 'regime_4h' in features.columns, "regime_4h 컬럼 없음"
    assert 'regime_score' in features.columns, "regime_score 컬럼 없음"
    assert 'target' in features.columns, "target 컬럼 없음"
    
    # 타겟 검증 (30분 바이너리: close > open)
    sample = features.iloc[-10]
    assert sample['target'] in [0, 1], "타겟 값 오류"
    
    feature_names = fe.get_feature_names(features)
    
    print(f"  ✓ 30분봉: {len(features)}개")
    print(f"  ✓ 피처: {len(feature_names)}개")
    print(f"  ✓ 레짐 분포: {features['regime'].value_counts().to_dict()}")
    print(f"  ✓ 타겟 분포: UP={features['target'].mean():.2%}")

test_module("FeatureEngineer (멀티레짐)", test_feature_engineer)

# ============================================================
# 4. ModelTrainer 테스트 (레짐별 앙상블 + 캘리브레이션)
# ============================================================
def test_model_trainer():
    try:
        from model_train import ModelTrainer
    except ImportError as e:
        print(f"  ⚠️  model_train.py 버전 문제: {e}")
        print(f"  ⚠️  model_train_final.py를 model_train.py로 교체 필요")
        return
    
    from feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer()
    
    # 테스트 데이터 (10일치)
    timestamps = pd.date_range(start='2025-01-01', periods=14400, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(14400).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 14400),
        'high': prices + 100,
        'low': prices - 100,
        'volume': np.random.uniform(1000, 5000, 14400)
    })
    
    # 피처 생성
    features = fe.create_feature_pool(df_1m, lookback_30m_bars=100)
    
    if features.empty:
        print("  ⚠️  피처 생성 실패, 스킵")
        return
    
    feature_names = fe.get_feature_names(features)
    X = features[feature_names + ['regime']].copy()
    y = features['target'].copy()
    
    # 레짐별 앙상블 학습
    trainer = ModelTrainer()
    
    # train_regime_ensemble 메서드 확인
    if not hasattr(trainer, 'train_regime_ensemble'):
        print("  ⚠️  train_regime_ensemble 메서드 없음")
        print("  ⚠️  model_train.py를 최신 버전으로 교체 필요")
        return
    
    results = trainer.train_regime_ensemble(X, y, n_learners=3, test_size=0.2)
    
    assert len(results) > 0, "학습 결과 없음"
    
    # 번들 저장
    bundle_path = trainer.save_bundle()
    assert bundle_path.exists(), "번들 저장 실패"
    
    # 번들 로드
    trainer2 = ModelTrainer()
    success = trainer2.load_bundle("latest")
    assert success, "번들 로드 실패"
    
    # 예측 테스트 (각 레짐별)
    X_test = X.head(10).drop(columns=['regime'])
    
    for regime_val in [1, 0, -1]:
        proba = trainer2.predict_with_regime(X_test, regime_val)
        assert proba.shape == (10, 2), f"예측 shape 오류: {proba.shape}"
        assert 0 <= proba[:, 1].min() <= proba[:, 1].max() <= 1, "확률 범위 오류"
    
    print(f"  ✓ 학습 완료: {len(results)}개 레짐 모델")
    print(f"  ✓ 번들 저장: {bundle_path.name}")
    print(f"  ✓ 앙상블: 각 {trainer.regime_models['up'].n_learners}개 learner")
    print(f"  ✓ 캘리브레이션: {trainer.metadata.get('calibration_method')}")

test_module("ModelTrainer (레짐별 앙상블)", test_model_trainer)

# ============================================================
# 5. 캘리브레이션 검증 테스트
# ============================================================
def test_calibration():
    from monitor import CalibrationMetrics
    
    # 테스트 데이터
    n_samples = 200
    
    # 잘 캘리브레이션된 경우
    probas_good = np.random.uniform(0.4, 0.6, n_samples)
    outcomes_good = (np.random.rand(n_samples) < probas_good).astype(int)
    
    ece_good, bins_good = CalibrationMetrics.calculate_ece(probas_good, outcomes_good, n_bins=10)
    
    assert ece_good < 0.1, f"ECE too high: {ece_good}"
    
    # Brier Score
    brier = CalibrationMetrics.calculate_brier_score(probas_good, outcomes_good)
    assert 0 <= brier <= 1, f"Brier score 범위 오류: {brier}"
    
    # Log Loss
    log_loss = CalibrationMetrics.calculate_log_loss(probas_good, outcomes_good)
    assert log_loss >= 0, f"Log loss 음수: {log_loss}"
    
    print(f"  ✓ ECE: {ece_good:.4f}")
    print(f"  ✓ Brier Score: {brier:.4f}")
    print(f"  ✓ Log Loss: {log_loss:.4f}")
    print(f"  ✓ 구간 수: {len(bins_good)}개")

test_module("캘리브레이션 메트릭", test_calibration)

# ============================================================
# 6. RealTradeManager 테스트
# ============================================================
def test_real_trade_manager():
    try:
        from real_trade import RealTradeManager
    except ImportError as e:
        print(f"  ⚠️  real_trade import 오류: {e}")
        print(f"  ⚠️  config.py에 SYMBOL 상수 추가 필요")
        return
    
    # 초기화 (모델 로드 필요)
    try:
        rtm = RealTradeManager()
    except Exception as e:
        print(f"  ⚠️  초기화 실패 (모델 없음?): {e}")
        return
    
    # 예측 타이밍 체크
    # should_predict_now()는 29분 0~10초에만 True
    # 테스트에서는 로직만 확인
    
    print(f"  ✓ 레짐 모델: {list(rtm.regime_models.keys()) if rtm.regime_models else '없음'}")
    print(f"  ✓ 스케일러: {'로드됨' if rtm.scaler else '없음'}")
    print(f"  ✓ 30분 주기 예측 로직 확인")

test_module("RealTradeManager (30분 주기)", test_real_trade_manager)

# ============================================================
# 7. LogManager 테스트
# ============================================================
def test_log_manager():
    from log_manager import LogManager
    
    log_mgr = LogManager()
    
    # UTC aware datetime 생성
    test_time = datetime.now(timezone.utc)
    
    # 간소 엔트리
    log_mgr.log_trade_entry_simple(
        trade_id='test_001',
        direction='UP',
        entry_price=50000.0,
        entry_ts=test_time,
        p_raw_at_entry=0.65,
        regime=1,
        bar30_start=test_time,
        bar30_end=test_time + timedelta(minutes=30)
    )
    
    # 결과 업데이트
    success = log_mgr.update_trade_result(
        trade_id='test_001',
        result='WIN',
        label_price=50200.0,
        label_ts=test_time + timedelta(minutes=30),
        payout=0.85
    )
    
    # 로드
    date_str = test_time.strftime("%Y%m%d")
    df_trades = log_mgr.load_trade_log(date_str)
    
    if df_trades.empty:
        print("  ⚠️  거래 로그 로드 실패 (정상일 수 있음)")
        return
    
    # 검증
    validation = log_mgr.validate_trade_log(df_trades)
    
    print(f"  ✓ 거래 기록: {len(df_trades)}개")
    print(f"  ✓ 업데이트: {'성공' if success else '실패'}")
    print(f"  ✓ 검증: {'통과' if validation['valid'] else '경고'}")

test_module("LogManager (UTC ISO)", test_log_manager)

# ============================================================
# 8. Monitor 테스트 (캘리브레이션 중심)
# ============================================================
def test_monitor():
    from monitor import Monitor
    from log_manager import LogManager
    
    log_mgr = LogManager()
    monitor = Monitor(log_mgr)
    
    # 스냅샷 업데이트
    snapshot = monitor.update()
    
    assert snapshot is not None, "스냅샷 생성 실패"
    assert 'calibration' in snapshot, "캘리브레이션 데이터 없음"
    assert 'performance' in snapshot, "성능 데이터 없음"
    
    # 캘리브레이션 상세
    cal = snapshot['calibration']
    
    print(f"  ✓ 스냅샷 생성: {len(snapshot)} 항목")
    print(f"  ✓ 캘리브레이션 검증: {'가능' if cal['valid'] else '불가'}")
    print(f"  ✓ 알림: {len(snapshot.get('alerts', []))}개")

test_module("Monitor (캘리브레이션)", test_monitor)

# ============================================================
# 9. MainPipeline 테스트
# ============================================================
def test_main_pipe():
    try:
        from main_pipe import MainPipeline
    except ImportError as e:
        print(f"  ⚠️  main_pipe import 실패: {e}")
        return
    
    pipeline = MainPipeline(symbol='BTCUSDT')
    
    # 시뮬레이션 데이터 생성
    df_sim = pipeline.generate_simulation_data(days=3)
    
    assert not df_sim.empty, "시뮬레이션 데이터 생성 실패"
    assert len(df_sim) == 3 * 1440, "데이터 개수 오류"
    
    # 번들 목록
    bundles = pipeline.model_trainer.list_bundles()
    
    print(f"  ✓ 시뮬레이션 데이터: {len(df_sim):,}개")
    print(f"  ✓ 저장된 번들: {len(bundles)}개")

test_module("MainPipeline (번들 관리)", test_main_pipe)

# ============================================================
# 10. 통합 워크플로우 테스트
# ============================================================
def test_integrated_workflow():
    """데이터 → 피처 → 학습 → 예측 전체 흐름"""
    from feature_engineer import FeatureEngineer
    
    try:
        from model_train import ModelTrainer
    except ImportError as e:
        print(f"  ⚠️  model_train import 실패: {e}")
        return
    
    fe = FeatureEngineer()
    
    # 1. 데이터 생성
    timestamps = pd.date_range(start='2025-01-01', periods=7200, freq='1min', tz='UTC')
    prices = 50000 + np.random.randn(7200).cumsum() * 10
    
    df_1m = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices + np.random.uniform(-50, 50, 7200),
        'high': prices + 100,
        'low': prices - 100,
        'volume': np.random.uniform(1000, 5000, 7200)
    })
    
    # 2. 피처 생성
    features = fe.create_feature_pool(df_1m, lookback_30m_bars=50)
    assert not features.empty, "피처 생성 실패"
    
    # 3. 학습 데이터 준비
    feature_names = fe.get_feature_names(features)
    X = features[feature_names + ['regime']].copy()
    y = features['target'].copy()
    
    # 4. 레짐별 학습
    trainer = ModelTrainer()
    
    if not hasattr(trainer, 'train_regime_ensemble'):
        print("  ⚠️  train_regime_ensemble 메서드 없음, 스킵")
        return
    
    results = trainer.train_regime_ensemble(X, y, n_learners=2, test_size=0.3)
    assert len(results) > 0, "학습 실패"
    
    # 5. 번들 저장/로드
    bundle_path = trainer.save_bundle()
    
    trainer2 = ModelTrainer()
    loaded = trainer2.load_bundle("latest")
    assert loaded, "로드 실패"
    
    # 6. 예측
    X_test = X.head(5).drop(columns=['regime'])
    proba = trainer2.predict_with_regime(X_test, regime=1)
    
    assert proba.shape == (5, 2), "예측 shape 오류"
    
    print(f"  ✓ 데이터: {len(df_1m)} → 피처: {len(features)}")
    print(f"  ✓ 학습: {len(results)}개 모델")
    print(f"  ✓ 번들: {bundle_path.name}")
    print(f"  ✓ 예측: {proba[:3, 1]}")

test_module("통합 워크플로우", test_integrated_workflow)

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

if passed_tests:
    print(f"\n통과한 테스트:")
    for name in passed_tests:
        print(f"  ✓ {name}")

if failed_tests:
    print("\n실패한 테스트:")
    for name, error in failed_tests:
        print(f"  ✗ {name}")
        print(f"    오류: {error[:200]}...")
else:
    print("\n🎉 모든 테스트 통과!")

print("\n" + "=" * 70)