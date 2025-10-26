"""
메인 파이프라인 - 전체 시스템 오케스트레이션
버전: 1.3.0
"""

import time
import schedule
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import signal
import sys
import numpy as np
import pandas as pd

import config
from real_trade import RealTrader
from model_train import ModelTrainer
from data_merge import DataMerger
from monitor import Monitor
from log_manager import LogManager

class MainPipeline:
    """
    메인 파이프라인
    - 실시간 거래 루프
    - 스케줄러 기반 백그라운드 태스크
    - 자동 재학습 (윌슨 하한 기반)
    - 일일 유지보수
    - 예외 처리 및 복구
    """
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        """
        초기화
        
        Args:
            symbol: 거래 심볼
        """
        self.symbol = symbol
        self.is_running = False
        
        # 컴포넌트 초기화
        self.trader = RealTrader(symbol=symbol)
        self.model_trainer = ModelTrainer()
        self.data_merger = DataMerger()
        self.log_manager = LogManager()
        self.monitor = Monitor(self.log_manager)
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("=" * 60)
        print(f"메인 파이프라인 초기화 완료: {symbol}")
        print(f"시스템 버전: {config.SYSTEM_VERSION}")
        print("=" * 60)
    
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
        
        # 2. 모델 로드
        if not self.trader.model_loaded:
            print("  ⚠️  모델 미로드, 학습된 모델 필요")
            
            # 초기 학습 필요 시 수행
            # success = self.retrain_pipeline()
            # if not success:
            #     return False
        else:
            print("  ✅ 모델 로드 완료")
        
        # 3. 로그 디렉토리 확인
        for dir_path in config.DIRS_TO_CREATE:
            if not dir_path.exists():
                print(f"  ❌ 디렉토리 없음: {dir_path}")
                return False
        
        print("  ✅ 디렉토리 구조 확인")
        
        # 4. 초기 모니터링
        self.monitor.print_summary()
        
        print("\n✅ 시스템 초기화 완료")
        return True
    
    def trading_loop(self) -> None:
        """거래 루프 (1분마다 실행)"""
        try:
            self.trader.run()
        except Exception as e:
            print(f"❌ 거래 루프 오류: {e}")
            # 복구 시도
            time.sleep(5)
    
    def scheduled_tasks(self) -> None:
        """스케줄러 기반 백그라운드 태스크"""
        # 모니터링 업데이트 (5분마다)
        schedule.every(config.MONITOR_UPDATE_INTERVAL).seconds.do(
            self._safe_execute, self.monitor.update
        )
        
        # 리포트 생성 (4시간마다)
        schedule.every(config.REPORT_INTERVAL).seconds.do(
            self._safe_execute, self.generate_report
        )
        
        # 재학습 체크 (1시간마다)
        schedule.every(3600).seconds.do(
            self._safe_execute, self.auto_retrain_check
        )
        
        # 일일 유지보수 (자정)
        schedule.every().day.at("00:00").do(
            self._safe_execute, self.daily_maintenance
        )
    
    def _safe_execute(self, func, *args, **kwargs) -> None:
        """안전 실행 래퍼 (예외 처리)"""
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"❌ 스케줄 태스크 실행 오류 ({func.__name__}): {e}")
    
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
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        print(f"\n📅 학습 기간: {start_date} ~ {end_date}")
        
        try:
            # 1. 데이터 병합 및 정제
            print("\n1️⃣  데이터 준비")
            X, y = self.data_merger.create_retrain_dataset(start_date, end_date)
            
            if X.empty or y.empty:
                print("❌ 데이터 준비 실패")
                return False
            
            print(f"  ✅ 데이터 준비 완료: X={X.shape}, y={y.shape}")
            
            # 2. 데이터 검증
            print("\n2️⃣  데이터 검증")
            valid, errors = self.data_merger.data_loader.validate_data_quality(X, y)
            
            if not valid:
                print("  ⚠️  데이터 품질 이슈:")
                for error in errors:
                    print(f"    - {error}")
                
                # 치명적 오류가 아니면 계속 진행
                if any('불일치' in e or '타입' in e for e in errors):
                    print("  ❌ 치명적 오류, 중단")
                    return False
            
            print("  ✅ 데이터 검증 통과")
            
            # 3. 모델 학습
            print("\n3️⃣  모델 학습")
            
            # 레짐별 모델 학습
            if 'regime' in X.columns:
                results = self.model_trainer.train_ensemble_regime(
                    X, y, regime_col='regime', test_size=0.2
                )
            else:
                results = self.model_trainer.train_single_model(X, y, test_size=0.2)
            
            print("\n  📊 학습 결과:")
            for model_name, result in results.items():
                print(f"    {model_name}: Accuracy={result.get('accuracy', 0):.4f}")
            
            # 4. 모델 저장
            print("\n4️⃣  모델 저장")
            self.model_trainer.save_models()
            
            # 5. 모델 재로드 (Trader에 적용)
            print("\n5️⃣  모델 재로드")
            self.trader.model_trainer.load_models()
            self.trader.model_loaded = True
            
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
        import numpy as np
        
        n = len(df_closed)
        wins = (df_closed['result'] == 'WIN').sum()
        p_hat = wins / n
        
        z = 1.96  # 95% 신뢰수준
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
        lower_bound = center - margin
        
        print(f"  승률: {p_hat:.2%}")
        print(f"  윌슨 하한: {lower_bound:.2%}")
        print(f"  임계값: {config.RETRAIN_WIN_RATE_THRESHOLD:.0%}")
        
        if lower_bound < config.RETRAIN_WIN_RATE_THRESHOLD:
            print("\n  ⚠️  재학습 필요!")
            
            # 재학습 실행
            success = self.retrain_pipeline()
            
            if success:
                print("  ✅ 재학습 완료")
                
                # 동적 필터 초기화
                self.trader.filter_state = self.trader._load_filter_state()
            else:
                print("  ❌ 재학습 실패")
        else:
            print("  ✅ 승률 양호, 재학습 불필요")
    
    def daily_maintenance(self) -> None:
        """일일 유지보수"""
        print("\n" + "=" * 60)
        print("🛠️  일일 유지보수 시작")
        print("=" * 60)
        
        # 1. 로그 로테이션
        print("\n1️⃣  로그 정리")
        # 구현: 오래된 로그 압축 또는 삭제
        
        # 2. 캐시 정리
        print("\n2️⃣  캐시 정리")
        self.data_merger.data_loader.clean_cache(older_than_days=7)
        
        # 3. 성능 리포트
        print("\n3️⃣  일일 성능 리포트")
        self.generate_report()
        
        # 4. 데이터 백업
        print("\n4️⃣  데이터 백업")
        # 구현: 중요 데이터 백업
        
        print("\n✅ 일일 유지보수 완료")
        print("=" * 60)
    
    def generate_report(self) -> None:
        """리포트 생성"""
        print("\n" + "=" * 60)
        print("📊 성능 리포트 생성")
        print("=" * 60)
        
        # 모니터링 요약 출력
        self.monitor.print_summary()
        
        # 트렌드 분석
        trend = self.monitor.get_performance_trend(hours=24)
        
        if trend['win_rate_trend']:
            import numpy as np
            print("\n📈 24시간 트렌드:")
            print(f"  평균 승률: {np.mean(trend['win_rate_trend']):.2%}")
            print(f"  평균 PnL: {np.mean(trend['pnl_trend']):.2f}")
            print(f"  평균 진입률: {np.mean(trend['entry_rate_trend']):.1f}회/시간")
        
        # 거래 상태
        status = self.trader.get_status()
        print("\n💼 거래 상태:")
        print(f"  활성 포지션: {status['active_positions']} / {status['max_positions']}")
        print(f"  동적 컷오프: {status['dynamic_cutoff']:.3f}")
        print(f"  필터 패턴: {status['filter_patterns']}개")
        
        print("\n" + "=" * 60)
    
    def start(self) -> None:
        """시스템 시작"""
        # 초기화
        if not self.initialize_system():
            print("❌ 시스템 초기화 실패")
            return
        
        # 스케줄 설정
        self.scheduled_tasks()
        
        self.is_running = True
        
        print("\n🚀 시스템 시작")
        print("  - Ctrl+C로 안전 종료")
        print("  - 거래 루프: 1분마다")
        print("  - 모니터링: 5분마다")
        print("  - 리포트: 4시간마다")
        print("=" * 60 + "\n")
        
        # 메인 루프
        try:
            while self.is_running:
                # 거래 루프 실행
                self.trading_loop()
                
                # 스케줄 태스크 실행
                schedule.run_pending()
                
                # 1분 대기
                time.sleep(60)
                
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
        
        # 1. 활성 포지션 정리
        print("\n1️⃣  활성 포지션 정리")
        if self.trader.active_positions:
            print(f"  활성 포지션 {len(self.trader.active_positions)}개 발견")
            # 실제 환경에서는 포지션 청산 필요
            print("  ⚠️  포지션 수동 정리 필요")
        else:
            print("  ✅ 활성 포지션 없음")
        
        # 2. 상태 저장
        print("\n2️⃣  상태 저장")
        self.trader._save_filter_state()
        print("  ✅ 필터 상태 저장 완료")
        
        # 3. 최종 리포트
        print("\n3️⃣  최종 리포트")
        self.generate_report()
        
        print("\n✅ 시스템 종료 완료")
        print("=" * 60)
    
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
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        X, y = self.data_merger.create_retrain_dataset(start_date, end_date)
        
        if X.empty or y.empty:
            print("❌ 데이터 준비 실패")
            return {}
        
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
        best_metrics = None
        
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
                                
                                # 학습
                                if 'regime' in X.columns:
                                    results = temp_trainer.train_ensemble_regime(
                                        X, y, regime_col='regime', test_size=0.2
                                    )
                                else:
                                    results = temp_trainer.train_single_model(
                                        X, y, test_size=0.2
                                    )
                                
                                # 원래 파라미터 복원
                                config.LIGHTGBM_PARAMS = original_params
                                
                                # 평균 정확도 계산
                                scores = [r.get('accuracy', 0) for r in results.values()]
                                avg_score = sum(scores) / len(scores) if scores else 0
                                
                                # 최고 점수 갱신
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = params
                                    best_metrics = results
                                    
                                    print(f"\n  [{tested}/{total_combinations}] ✨ 신기록!")
                                    print(f"    Accuracy: {avg_score:.4f}")
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
            print(f"  - 평균 Accuracy: {best_score:.4f}")
            
            if best_metrics:
                print(f"\n  레짐별 성능:")
                for regime_name, metrics in best_metrics.items():
                    print(f"    {regime_name}: {metrics.get('accuracy', 0):.4f}")
            
            # 설정 파일에 저장 (선택)
            print(f"\n💾 최적 파라미터를 적용하시겠습니까? (y/n): ", end="")
            # 자동으로 적용하지 않고 반환만
            
        else:
            print("\n❌ 더 나은 파라미터를 찾지 못했습니다.")
        
        print("\n" + "=" * 60)
        
        return best_params or {}
    
    def generate_simulation_data(self, days: int = 90) -> pd.DataFrame:
        """
        시뮬레이션 데이터 생성
        
        Args:
            days: 생성할 일수
        
        Returns:
            시뮬레이션 1분봉 데이터
        """
        print(f"\n📊 시뮬레이션 데이터 생성 ({days}일)")
        
        n_minutes = days * 1440  # 하루 1440분
        
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_minutes)]
        
        # 랜덤 워크
        np.random.seed(42)
        base_price = 50000.0
        returns = np.random.normal(0, 0.002, n_minutes)
        prices = base_price * (1 + returns).cumprod()
        
        data = []
        for i, ts in enumerate(timestamps):
            close = prices[i]
            open_price = close + np.random.uniform(-50, 50)
            high = max(open_price, close) + np.random.uniform(0, 100)
            low = min(open_price, close) - np.random.uniform(0, 100)
            volume = np.random.uniform(1000, 5000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        print(f"  ✅ 생성 완료: {len(df):,}개 데이터")
        
        return df
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        print(f"\n⚠️  시그널 수신: {signum}")
        self.is_running = False


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("바이너리 옵션 트레이딩 시스템")
    print(f"버전: {config.SYSTEM_VERSION}")
    print("=" * 60)
    
    # 파이프라인 초기화
    pipeline = MainPipeline(symbol='BTCUSDT')
    
    # 시스템 시작
    pipeline.start()