"""
메인 파이프라인 - 전체 시스템 오케스트레이션
버전: 1.4.0

핵심 기능:
- 30분 바이너리 옵션 실시간 거래
- 자동 재학습 (윌슨 하한 기반)
- 하이퍼파라미터 튜닝
- 레짐별 앙상블 학습
- 캘리브레이션 모니터링
"""

import time
import schedule
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import signal
import sys
import numpy as np
import pandas as pd
import threading
import os, json

import config
from real_trade import RealTradeManager
from model_train import ModelTrainer
from log_manager import LogManager
from feature_engineer import FeatureEngineer
from data_loader import DataLoader
from monitor import Monitor
from timeframe_manager import TimeframeManager


STATE_PATH = os.path.join("state", "initial_start_date.json")

def _persist_initial_start_date(start_str: str):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"start_date": start_str}, f)

def _load_initial_start_date() -> str | None:
    """저장된 start_date 불러오기"""
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("start_date")
    except Exception:
        return None


class MainPipeline:
    """
    메인 파이프라인
    - 실시간 거래 (30분 주기)
    - 백그라운드 모니터링
    - 자동 재학습 (윌슨 하한)
    - 하이퍼파라미터 최적화
    - 레짐별 앙상블 학습
    """
    
    def __init__(self, cfg=None, data_loader=None, feature_engineer=None, symbol: str = 'BTCUSDT'):
        """
        초기화
        
        Args:
            symbol: 거래 심볼
        """
        self.symbol = symbol
        self.is_running = False
        self.cfg = cfg or config
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        # 컴포넌트 초기화
        self.trader = RealTradeManager()
        self.model_trainer = ModelTrainer()
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.log_manager = LogManager()
        self.monitor = Monitor(self.log_manager)
        
        # 백그라운드 스레드
        self.monitor_thread = None
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("=" * 60)
        print(f"메인 파이프라인 초기화 완료: {symbol}")
        print(f"시스템 버전: {config.SYSTEM_VERSION}")
        print("=" * 60)


    def generate_simulation_data(
    self,
    days: int = 180,
    start_date: str | None = None,
    end_date: str | None = None,
    symbol: str | None = None,
    timeframe: str = "30m",
    ) -> pd.DataFrame:
        """
        실제 1분봉 데이터를 로드해 초기 구동/차트용 캔들을 반환.
        반환 형식: [timestamp, open, high, low, close, volume] (UTC)
        """

        # 0️⃣ 기본값 세팅
        yesterday_utc = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        end_date = (end_date or yesterday_utc.strftime("%Y-%m-%d"))

        # start_date는 처음 한 번만 기억
        remembered = _load_initial_start_date()
        if remembered:
            start_date = remembered
        else:
            if not start_date:
                start_dt = (datetime.strptime(end_date, "%Y-%m-%d").date() - timedelta(days=days))
                start_date = start_dt.strftime("%Y-%m-%d")
            _persist_initial_start_date(start_date)

        # UTC 안전성 확보
        end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
        start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_d > yesterday_utc:
            end_d = yesterday_utc
            end_date = end_d.strftime("%Y-%m-%d")
        if start_d > end_d:
            start_d = end_d - timedelta(days=max(1, days))
            start_date = start_d.strftime("%Y-%m-%d")

        sym = symbol or getattr(self.cfg, "DEFAULT_SYMBOL", "BTCUSDT")

        if self.data_loader is None:
            raise RuntimeError("data_loader가 설정되지 않았습니다.")

        # 1️⃣ 1분봉 데이터 로드
        df_1m = self.data_loader.load_price_data(
            start_date=start_date,
            end_date=end_date,
            symbol=sym,
        )
        if df_1m is None or df_1m.empty:
            raise RuntimeError(f"실제 데이터 로드 실패 또는 빈 데이터: {sym} {start_date}~{end_date}")

        # 2️⃣ 타임프레임 집계
        tfm = TimeframeManager()
        tf = timeframe.lower().replace("min", "m")  # "30min" 등 입력도 허용

        if tf in ("30m", "30"):
            df_tf = tfm.aggregate_1m_to_30m(df_1m, realtime_safe=False)
            if df_tf is None or df_tf.empty:
                raise RuntimeError("30분봉 집계 결과가 비어 있습니다.")

            # bar30_start → timestamp 로 통일
            if "bar30_start" not in df_tf.columns:
                raise RuntimeError("집계 결과에 bar30_start 컬럼이 없습니다.")
            df_tf = df_tf.rename(columns={"bar30_start": "timestamp"})

            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            return df_tf[cols].copy()

        else:
            # 범용 리샘플 (예: 15m, 1h 등)
            df_res = tfm.resample_to_timeframe(df_1m, tf)
            if df_res is None or df_res.empty:
                raise RuntimeError(f"{timeframe} 집계 결과가 비어 있습니다.")

            start_col = f"bar_{tf}_start"
            if start_col not in df_res.columns:
                raise RuntimeError(f"집계 결과에 {start_col} 컬럼이 없습니다.")
            df_res = df_res.rename(columns={start_col: "timestamp"})

            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            return df_res[cols].copy()
    
    def initialize_system(self) -> bool:
        """
        시스템 초기화
        
        Returns:
            초기화 성공 여부
        """
        print("\n🔧 시스템 초기화 중...")
        
        # 1. Config 검증
        valid, errors = config.validate_config()
        if not valid:
            print("❌ Config 검증 실패:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("  ✅ Config 검증 통과")
        
        # 2. 레짐 번들 로드 시도
        print("  🔎 최신 번들(latest) 로드 시도")
        if not self.model_trainer.load_bundle("latest"):
            print("  ⚠️  최신 번들이 없거나 손상됨 → 초기 학습 필요")
            response = input("  초기 학습을 진행하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                success = self.initial_training()
                if not success:
                    print("  ❌ 초기 학습 실패")
                    return False
            else:
                print("  ❌ 모델 없이 실행 불가")
                return False
        else:
            print("  ✅ 번들 로드 완료")
            # 트레이더 쪽에도 같은 트레이너/상태 전달
            self.trader.trainer = self.model_trainer
            setattr(self.trader, "model_loaded", True)
                
        # 3. 로그 디렉토리 확인
        for dir_path in config.DIRS_TO_CREATE:
            if not dir_path.exists():
                print(f"  ❌ 디렉토리 없음: {dir_path}")
                return False
        
        print("  ✅ 디렉토리 구조 확인")
        
        # 4. 초기 모니터링
        print("\n📊 초기 모니터링:")
        self.monitor.print_summary()
        
        print("\n✅ 시스템 초기화 완료")
        return True
    
    def initial_training(self, start_date: str = "2025-01-01") -> bool:
        """
        초기 학습 (실제 데이터 기반)
        
        Args:
            start_date: 학습 시작일 (UTC 기준)
        
        Returns:
            학습 성공 여부
        """
        print("\n" + "=" * 60)
        print("🎓 초기 학습 시작 (실제 데이터 기반)")
        print("=" * 60)

        # 1️⃣ 실제 가격 데이터 로드
        print("\n1️⃣  가격 데이터 로드")
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            df_1m = self.data_loader.load_price_data(
                start_date=start_date,
                end_date=end_date,
                symbol=config.DEFAULT_SYMBOL
            )
        except Exception as e:
            print(f"  ❌ 데이터 로드 실패: {e}")
            return False

        if df_1m.empty:
            print("  ❌ 데이터가 비어 있습니다.")
            return False

        print(f"  ✅ 데이터 확보: {len(df_1m):,}개 ({start_date} ~ {end_date})")

        # 2️⃣ 특성 생성
        print("\n2️⃣  피처 생성 중...")
        df_feat = self.feature_engineer.create_feature_pool(df_1m)

        if df_feat.empty:
            print("  ❌ 피처 생성 실패")
            return False

        feature_names = self.feature_engineer.get_feature_names(df_feat)
        print(f"  ✅ 피처 {len(feature_names)}개, 샘플 {len(df_feat):,}개")

        # 3️⃣ X, y 분리 (+ regime 포함 보장)
        X = df_feat[feature_names].copy()
        y = df_feat["target"].copy()

        if "regime" in df_feat.columns and "regime" not in X.columns:
            X = X.join(df_feat["regime"])

        print("▶ pre-train (initial_training): regime counts =",
            X.get("regime", pd.Series(dtype="int64")).value_counts(dropna=False).to_dict())

        # 4️⃣ 레짐별 앙상블 학습
        print("\n3️⃣  레짐별 앙상블 학습")
        success = self.train_regime_ensemble(X, y)

        if not success:
            print("  ❌ 학습 실패")
            return False

        # 5️⃣ 모델 번들 저장
        print("\n✅ 초기 학습 완료")
        print("=" * 60)
        return True

    
    def train_regime_ensemble(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        레짐별 앙상블 학습 (호출자는 X에 regime 포함 보장)
        """
        try:
            # 1) regime 포함 보장
            if 'regime' not in X.columns:
                # 가능하면 FeatureEngineer가 만든 df_feat에서 regime을 가져오도록 하세요.
                # 이 함수의 호출부(retrain_pipeline / initial_training)에서 X 만들기 직후
                # df_feat['regime']를 join해서 넘겨주는 방식이 가장 안전합니다.
                print("  ⚠️  X에 'regime' 없음 → 단일 레짐(UP=1)으로 가정")
                X = X.copy()
                X['regime'] = 1

            # 2) 분포 확인(디버그용)
            print("▶ pre-train (pipeline): regime counts =",
                X['regime'].value_counts(dropna=False).to_dict())

            # 3) 학습(내부에서 레짐별 분리 + 앙상블 + 캘리브레이션)
            results = self.model_trainer.train_model(
                X, y,
                n_learners=getattr(config, "N_ENSEMBLE", 3),
                test_size=0.2
            )

            # 4) 번들 저장 및 최신 링크/복사
            self.model_trainer.save_bundle()

            # 5) 바로 재로딩(무결성 체크)
            self.model_trainer.load_bundle("latest")

            # 6) 요약 출력(있을 때만)
            if isinstance(results, dict) and results:
                print("\n  ✅ 레짐별 학습 결과 요약:")
                for regime_name, metric in results.items():
                    if isinstance(metric, dict) and 'accuracy' in metric:
                        print(f"    - {regime_name}: acc={metric['accuracy']:.4f}, auc={metric.get('roc_auc', 0):.4f}")

            return True

        except Exception as e:
            print(f"  ❌ 학습 실패: {e}")
            return False

    
    def retrain_pipeline(self, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> bool:
        """
        재학습 파이프라인
        
        Args:
            start_date: 시작일 (None이면 최근 30일)
            end_date: 종료일 (None이면 오늘)
        
        Returns:
            재학습 성공 여부
        """
        print("\n" + "=" * 60)
        print("🔄 재학습 파이프라인 시작")
        print("=" * 60)
        
        # 날짜 설정
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if start_date is None:
            start_dt = datetime.now(timezone.utc) - timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        print(f"\n📅 학습 기간: {start_date} ~ {end_date}")
        
        try:
            # 1. 데이터 로드
            print("\n1️⃣  데이터 로드")
            df_1m = self.data_loader.load_price_data()
            
            if df_1m.empty:
                print("  ❌ 데이터 로드 실패")
                return False
            
            print(f"  ✅ 데이터 로드: {len(df_1m):,}개")
            
            # 2. 특성 생성
            print("\n2️⃣  특성 생성")
            df_feat = self.feature_engineer.create_feature_pool(df_1m)
            
            if df_feat.empty:
                print("  ❌ 특성 생성 실패")
                return False
            
            print(f"  ✅ 특성 생성: {len(df_feat)}개")
            
            # 3. X, y 분리
            feature_names = self.feature_engineer.get_feature_names(df_feat)
            X = df_feat[feature_names].copy()
            y = df_feat['target'].copy()

            # 🔎 regime/ADX 스냅샷 (라벨 생성이 문제인지 즉시 판별)
            print("▶ post-feature: regime counts =",
                df_feat.get("regime", pd.Series(dtype='int64')).value_counts(dropna=False).to_dict())

            adx_cols = [c for c in df_feat.columns if 'adx' in c.lower()][:10]
            print("▶ adx cols (sample):", adx_cols)
            if adx_cols:
                print(df_feat[adx_cols].quantile([0, 0.1, 0.5, 0.9, 1]).T)

            # 학습에 넘기는 X에 regime 없으면 잠시 붙여서 분포 재확인
            if "regime" not in X.columns and "regime" in df_feat.columns:
                X = X.join(df_feat["regime"])
                print("▶ pre-train: regime counts (X) =",
                    X.get("regime", pd.Series(dtype='int64')).value_counts(dropna=False).to_dict())
            
            # 4. 데이터 검증
            print("\n3️⃣  데이터 검증")
            valid, errors, warnings = self.feature_engineer.validate_features(df_feat)
            
            if not valid:
                print("  ⚠️  데이터 품질 이슈:")
                for error in errors:
                    print(f"    - {error}")
                
                # 치명적 오류가 아니면 계속 진행
                if any('타겟' in e or 'Inf' in e for e in errors):
                    print("  ❌ 치명적 오류, 중단")
                    return False
            
            if warnings:
                print("  ⚠️  경고:")
                for warning in warnings:
                    print(f"    - {warning}")
            
            print("  ✅ 데이터 검증 통과")
            
            # 5. 레짐별 앙상블 학습
            print("\n4️⃣  레짐별 앙상블 학습")
            success = self.train_regime_ensemble(X, y)
            
            if not success:
                print("  ❌ 학습 실패")
                return False
            
            # 6. 모델 재로드 (Trader에 적용)
            print("\n5️⃣  모델 재로드")
            ok = self.model_trainer.load_bundle("latest")
            if not ok:
                print("  ❌ 번들 로드 실패")
                return False

            # Trader가 번들을 직접 쓰도록 연결(트레이더 인터페이스에 맞춰 아래 중 하나 택1)
            if hasattr(self.trader, "attach_trainer"):
                # 가장 깔끔한 방식: 트레이너 전체를 붙여 추론 시 trainer.predict_with_regime(...) 호출
                self.trader.attach_trainer(self.model_trainer)
            elif hasattr(self.trader, "set_model_bundle"):
                self.trader.set_model_bundle(
                    regime_models=self.model_trainer.regime_models,
                    scaler=self.model_trainer.scaler,
                    metadata=self.model_trainer.metadata
                )
            else:
                # 최소 호환: 속성 직접 주입
                self.trader.regime_models = self.model_trainer.regime_models
                self.trader.scaler = self.model_trainer.scaler
                self.trader.model_trainer = self.model_trainer
                self.trader.model_loaded = True  # 트레이더가 이 플래그를 체크한다면
            
            print("\n✅ 재학습 파이프라인 완료")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 재학습 파이프라인 실패: {e}")
            print("=" * 60)
            return False
    
    def auto_retrain_check(self) -> None:
        """자동 재학습 체크 (윌슨 하한 기반)"""
        print("\n🔍 자동 재학습 체크")
        
        # 최근 거래 로드
        df_recent = self.log_manager.load_recent_trades(n=config.MIN_TRADES_FOR_RETRAIN)
        
        if len(df_recent) < config.MIN_TRADES_FOR_RETRAIN:
            print(f"  ⏳ 거래 수 부족 ({len(df_recent)} < {config.MIN_TRADES_FOR_RETRAIN})")
            return
        
        df_closed = df_recent[df_recent['status'] == 'CLOSED']
        
        if len(df_closed) < config.MIN_TRADES_FOR_RETRAIN:
            print(f"  ⏳ 완료 거래 부족 ({len(df_closed)} < {config.MIN_TRADES_FOR_RETRAIN})")
            return
        
        # 윌슨 하한 계산
        n = len(df_closed)
        wins = (df_closed['result'] == 'WIN').sum()
        p_hat = wins / n
        
        z = 1.96  # 95% 신뢰수준
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
        lower_bound = center - margin
        
        print(f"  거래 수: {n}건")
        print(f"  승률: {p_hat:.2%}")
        print(f"  윌슨 하한: {lower_bound:.2%}")
        print(f"  임계값: {config.RETRAIN_WIN_RATE_THRESHOLD:.0%}")
        
        if lower_bound < config.RETRAIN_WIN_RATE_THRESHOLD:
            print("\n  ⚠️  재학습 필요!")
            
            # 재학습 실행
            success = self.retrain_pipeline()
            
            if success:
                print("  ✅ 재학습 완료")
            else:
                print("  ❌ 재학습 실패")
        else:
            print("  ✅ 승률 양호, 재학습 불필요")
    
    def optimize_hyperparameters(self, lookback_days: int = 90) -> Dict:
        """
        하이퍼파라미터 최적화 (Grid Search)
        
        Args:
            lookback_days: 학습 데이터 일수
        
        Returns:
            최적 파라미터 딕셔너리
        """
        print("\n" + "=" * 60)
        print("🔍 하이퍼파라미터 최적화 시작")
        print("=" * 60)
        
        # 데이터 준비
        print("\n📊 학습 데이터 준비")
        
        df_1m = self.data_loader.load_price_data()
        
        if df_1m.empty:
            print("❌ 데이터 로드 실패")
            return {}
        
        df_feat = self.feature_engineer.create_feature_pool(df_1m)
        
        if df_feat.empty:
            print("❌ 특성 생성 실패")
            return {}
        
        feature_names = self.feature_engineer.get_feature_names(df_feat)
        X = df_feat[feature_names].copy()
        y = df_feat['target'].copy()
        
        print(f"  ✅ 데이터: X={X.shape}, y={y.shape}")
        
        # 파라미터 그리드
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.8, 0.9],
            'bagging_fraction': [0.7, 0.8, 0.9],
            'max_depth': [5, 7, 9]
        }
        
        print(f"\n🔎 탐색 공간:")
        total_combinations = 1
        for key, values in param_grid.items():
            print(f"  - {key}: {values}")
            total_combinations *= len(values)
        
        print(f"\n  총 조합: {total_combinations}개")
        
        # Grid Search
        best_params = None
        best_score = -1e9
        
        print(f"\n⚙️  최적화 진행 중...")
        
        tested = 0
        
        for num_leaves in param_grid['num_leaves']:
            for learning_rate in param_grid['learning_rate']:
                for feature_fraction in param_grid['feature_fraction']:
                    for bagging_fraction in param_grid['bagging_fraction']:
                        for max_depth in param_grid['max_depth']:
                            tested += 1
                            
                            # 파라미터 설정
                            params = config.LIGHTGBM_PARAMS.copy()
                            params.update({
                                'num_leaves': num_leaves,
                                'learning_rate': learning_rate,
                                'feature_fraction': feature_fraction,
                                'bagging_fraction': bagging_fraction,
                                'max_depth': max_depth
                            })
                            
                            try:
                                # 임시 트레이너 생성
                                temp_trainer = ModelTrainer()
                                
                                # 임시로 파라미터 변경
                                original_params = config.LIGHTGBM_PARAMS.copy()
                                config.LIGHTGBM_PARAMS = params
                                
                                # regime 제거 후 학습
                                X_train = X.drop(columns=['regime'], errors='ignore')
                                result = temp_trainer.train_model(X_train, y)
                                
                                # 원래 파라미터 복원
                                config.LIGHTGBM_PARAMS = original_params
                                
                                # 점수 평가
                                score = result['test_accuracy']
                                
                                # 최고 점수 갱신
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    
                                    print(f"\n  [{tested}/{total_combinations}] ✨ 신기록!")
                                    print(f"    Accuracy: {score:.4f}")
                                    print(f"    Params: leaves={num_leaves}, lr={learning_rate}, "
                                          f"feat={feature_fraction}, bag={bagging_fraction}, depth={max_depth}")
                                
                                elif tested % 10 == 0:
                                    print(f"  [{tested}/{total_combinations}] 진행 중... (최고: {best_score:.4f})")
                                
                            except Exception as e:
                                print(f"  [{tested}/{total_combinations}] ⚠️  실패: {e}")
                                continue
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("최적화 완료!")
        print("=" * 60)
        
        if best_params:
            print(f"\n🏆 최적 파라미터:")
            for key, value in best_params.items():
                print(f"  - {key}: {value}")
            
            print(f"\n📊 최고 성능:")
            print(f"  - Accuracy: {best_score:.4f}")
            
            print(f"\n💾 config.py의 LIGHTGBM_PARAMS를 수동으로 업데이트하세요")
            
        else:
            print("\n❌ 더 나은 파라미터를 찾지 못했습니다.")
        
        print("\n" + "=" * 60)
        
        return best_params or {}

    
    def monitor_loop(self) -> None:
        """모니터링 백그라운드 루프"""
        print("📊 모니터링 루프 시작")
        
        while self.is_running:
            try:
                # 모니터링 업데이트
                self.monitor.update()
                
                # 캘리브레이션 검증
                cal = self.monitor.perf_monitor.validate_calibration(window=200)
                
                if cal['valid'] and not cal['is_well_calibrated']:
                    print("\n⚠️  캘리브레이션 불량 감지!")
                    print(f"  ECE: {cal['metrics']['ece']:.4f}")
                    print(f"  갭: {cal['metrics']['calibration_gap']:.4f}")
                
                # 5분 대기
                time.sleep(config.MONITOR_UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"❌ 모니터링 에러: {e}")
                time.sleep(60)
    
    def start(self) -> None:
        """시스템 시작"""
        # 초기화
        if not self.initialize_system():
            print("❌ 시스템 초기화 실패")
            return
        
        self.is_running = True
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("\n🚀 시스템 시작")
        print("  - Ctrl+C로 안전 종료")
        print("  - 거래: 30분 주기 (29분 예측, 30분 진입)")
        print("  - 모니터링: 5분마다")
        print("  - 양방향 진입: UP/DOWN")
        print("=" * 60 + "\n")
        
        # 실시간 거래 시작 (블로킹)
        try:
            self.trader.run_live()
                
        except KeyboardInterrupt:
            print("\n⚠️  사용자 중단")
        except Exception as e:
            print(f"\n❌ 시스템 오류: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """시스템 안전 종료"""
        print("\n" + "=" * 60)
        print("🛑 시스템 종료 중...")
        print("=" * 60)
        
        self.is_running = False
        
        # 모니터링 스레드 종료 대기
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("\n1️⃣  모니터링 스레드 종료 대기...")
            self.monitor_thread.join(timeout=5)
            print("  ✅ 모니터링 종료")
        
        # 최종 리포트
        print("\n2️⃣  최종 리포트")
        self.monitor.print_summary()
        
        print("\n✅ 시스템 종료 완료")
        print("=" * 60)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        print(f"\n⚠️  시그널 수신: {signum}")
        self.is_running = False


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    parser = argparse.ArgumentParser(description='30분 바이너리 옵션 트레이딩 시스템')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'optimize'],
                        default='live', help='실행 모드')
    parser.add_argument('--symbol', default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--days', type=int, default=30, help='학습/백테스트 기간 (일)')
    parser.add_argument('--start', type=str, default=None, help='시작일(YYYY-MM-DD), 지정시 days 무시')
    parser.add_argument('--end', type=str, default=None, help='종료일(YYYY-MM-DD), 기본: 오늘(UTC)')
    args = parser.parse_args()

    print("=" * 60)
    print("바이너리 옵션 트레이딩 시스템")
    print(f"버전: {config.SYSTEM_VERSION}")
    print(f"모드: {args.mode}")
    print("=" * 60)

    # 파이프라인 초기화
    pipeline = MainPipeline(symbol=args.symbol)

    if args.mode == 'live':
        # 실시간 거래
        pipeline.start()

    elif args.mode == 'backtest':
        # -----------------------------
        # 백테스트 분기 (실데이터 기반)
        # -----------------------------
        print("\n📊 백테스트 모드")

        # 1) 기간 계산 (start/end 인자 우선, 없으면 days로 계산)
        if args.end is None:
            end_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            end_utc_str = args.end

        if args.start is None:
            start_utc_str = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
        else:
            start_utc_str = args.start

        # 2) 실데이터 로드 (일자별 CSV 자동 병합/다운로드)
        dl = DataLoader()
        df = dl.load_price_data(start_date=start_utc_str, end_date=end_utc_str, symbol=args.symbol)

        if df.empty:
            print("❌ 데이터 로드 실패 (빈 데이터)")
            sys.exit(1)

        # 3) BacktestManager 준비
        try:
            from real_trade import BacktestManager
            btm = BacktestManager()
        except Exception as e:
            print(f"❌ BacktestManager 초기화 실패: {e}")
            # 필요시 간이 백테스터로 대체 가능:
            # from real_trade import SimpleBacktester
            # btm = SimpleBacktester()
            sys.exit(1)

        # 4) 백테스트 실행
        result = btm.run_backtest(df)

        if isinstance(result, pd.DataFrame) and not result.empty:
            output_path = 'backtest_result_30m.csv'
            result.to_csv(output_path, index=False)
            print(f"\n✅ 결과 저장: {output_path}")
        else:
            print("⚠️ 백테스트 결과가 비었습니다.")

    elif args.mode == 'train':
        # 재학습 (실데이터 최신화 포함)
        print("\n🎓 재학습 모드")
        ok = pipeline.retrain_pipeline()
        if not ok:
            sys.exit(1)

    elif args.mode == 'optimize':
        # 하이퍼파라미터 최적화
        print("\n🔍 하이퍼파라미터 최적화 모드")
        best_params = pipeline.optimize_hyperparameters(lookback_days=args.days)

        if best_params:
            print("\n📝 config.py 업데이트 가이드:")
            print("```python")
            print("LIGHTGBM_PARAMS = {")
            for key, value in best_params.items():
                if isinstance(value, str):
                    print(f"    '{key}': '{value}',")
                else:
                    print(f"    '{key}': {value},")
            print("}")
            print("```")
    else:
        print(f"❌ 알 수 없는 모드: {args.mode}")
        sys.exit(1)