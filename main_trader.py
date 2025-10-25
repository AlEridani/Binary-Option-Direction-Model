
import os
import sys
import time
import signal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import traceback

from config import Config
from timeframe_manager import TimeframeManager
from feature_engineer import FeatureEngineer
from model_train import ModelTrainer
from trading_engine import TradingEngine
from log_manager import LogManager
from version_manager import VersionManager
from validator import Validator


class MainTrader:
    """메인 트레이딩 시스템"""
    
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.running = False
        
        # 설정
        self.config = Config
        
        # 컴포넌트 초기화
        self.tf_manager = TimeframeManager()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(Config)
        self.trading_engine = TradingEngine(self.model_trainer, symbol)
        self.log_manager = LogManager()
        self.version_manager = VersionManager()
        self.validator = Validator()
        
        # 1분봉 버퍼 (최근 N개 유지)
        self.buffer_size = 5000  # 약 3.5일
        self.df_1m_buffer = pd.DataFrame()
        
        # 마지막 처리 시각
        self.last_process_time = None
        
        # 재학습 관련
        self.retrain_in_progress = False
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"\n{'='*60}")
        print(f"30분봉 바이너리 옵션 트레이딩 시스템 초기화")
        print(f"{'='*60}")
        print(f"심볼: {self.symbol}")
        print(f"업데이트 주기: {self.config.UPDATE_INTERVAL_SECONDS}초")
        print(f"재학습 트리거: {self.config.RETRAIN_CHECK_INTERVAL}번 거래 후 승률 < {self.config.RETRAIN_MIN_WIN_RATE:.0%}")
        print(f"{'='*60}\n")
    
    def _signal_handler(self, signum, frame):
        """종료 시그널 처리"""
        print(f"\n\n종료 시그널 수신: {signum}")
        self.stop()
    
    # ==========================================
    # 초기화
    # ==========================================
    
    def initialize(self):
        """시스템 초기화"""
        print("시스템 초기화 중...")
        
        # 디렉토리 생성
        self.config.create_directories()
        self.config.validate_config()
        
        # 모델 로드 시도
        model_loaded = self.model_trainer.load_model()
        
        if not model_loaded:
            print("\n⚠️ 기존 모델 없음 - 초기 학습 필요")
            print("  초기 학습을 먼저 실행하세요:")
            print("  python initial_training.py")
            return False
        
        print("✓ 모델 로드 완료")
        
        # 성능 추적 상태 로드
        self.model_trainer.load_performance_state()
        
        # 마지막 거래 로그 로드 (재시작 시 상태 복원)
        today = datetime.now(self.config.LOG_TIMEZONE).strftime("%Y%m%d")
        last_trades = self.log_manager.load_trade_log(today)
        
        if len(last_trades) > 0:
            print(f"✓ 오늘 거래 내역: {len(last_trades)}건")
        
        print("✓ 시스템 초기화 완료\n")
        return True
    
    # ==========================================
    # 데이터 수집 (실제로는 API 연동)
    # ==========================================
    
    def fetch_latest_data(self) -> pd.DataFrame:
        """
        최신 1분봉 데이터 수집
        실제 구현 시: Binance API 등 연동
        
        Returns:
        --------
        DataFrame: 최신 1분봉 (1개)
        """
        # TODO: 실제 API 연동
        # from binance_api import BinanceAPIClient
        # api = BinanceAPIClient()
        # latest = api.get_latest_kline(symbol=self.symbol, interval='1m')
        
        # 현재는 시뮬레이션 (랜덤 데이터)
        now = datetime.now(timezone.utc)
        now = now.replace(second=0, microsecond=0)
        
        base_price = 42000 if len(self.df_1m_buffer) == 0 else self.df_1m_buffer.iloc[-1]['close']
        base_price += np.random.randn() * 20
        
        o = base_price + np.random.uniform(-20, 20)
        c = base_price + np.random.uniform(-20, 20)
        h = max(o, c) + np.random.uniform(0, 30)
        l = min(o, c) - np.random.uniform(0, 30)
        v = np.random.uniform(100, 1000)
        
        return pd.DataFrame([{
            'timestamp': now,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v
        }])
    
    def update_buffer(self, new_data: pd.DataFrame):
        """1분봉 버퍼 업데이트"""
        if len(self.df_1m_buffer) == 0:
            self.df_1m_buffer = new_data
        else:
            self.df_1m_buffer = pd.concat([self.df_1m_buffer, new_data], ignore_index=True)
        
        # 오래된 데이터 제거
        if len(self.df_1m_buffer) > self.buffer_size:
            self.df_1m_buffer = self.df_1m_buffer.iloc[-self.buffer_size:]
        
        # 타임스탬프 정렬
        self.df_1m_buffer = self.df_1m_buffer.sort_values('timestamp').reset_index(drop=True)
    
    # ==========================================
    # 메인 처리 루프
    # ==========================================
    
    def process_tick(self) -> bool:
        """
        1분마다 실행되는 틱 처리
        
        Returns:
        --------
        bool: 처리 성공 여부
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # 1. 최신 데이터 수집
            latest_data = self.fetch_latest_data()
            self.update_buffer(latest_data)
            
            # 버퍼가 충분한지 확인
            if len(self.df_1m_buffer) < 200:
                print(f"  버퍼 부족: {len(self.df_1m_buffer)}/200")
                return True
            
            # 2. 30분봉 변환
            df_30m = self.tf_manager.aggregate_1m_to_30m(self.df_1m_buffer)
            
            if len(df_30m) < 100:
                print(f"  30분봉 부족: {len(df_30m)}/100")
                return True
            
            # 3. 30분봉 close 시점인지 확인
            latest_bar = df_30m.iloc[-1]
            bar30_end = pd.to_datetime(latest_bar['bar30_end'], utc=True)
            
            # 현재 시각이 30분봉 close와 일치하는지 (±1분 허용)
            time_diff = abs((current_time - bar30_end).total_seconds())
            
            if time_diff > 60:
                # 30분봉 close 시점 아님
                return True
            
            # 이미 처리한 바인지 확인
            if self.last_process_time == bar30_end:
                return True
            
            print(f"\n{'='*60}")
            print(f"30분봉 close 시점 감지: {bar30_end}")
            print(f"{'='*60}")
            
            # 4. 피처 생성
            print("피처 생성 중...")
            features = self.feature_engineer.create_feature_pool(
                self.df_1m_buffer, 
                lookback_bars=100
            )
            
            if len(features) == 0:
                print("⚠️ 피처 생성 실패")
                return True
            
            # 30분봉에 피처 병합
            df_30m_with_features = df_30m.iloc[-len(features):].reset_index(drop=True)
            for col in features.columns:
                if col not in df_30m_with_features.columns:
                    df_30m_with_features[col] = features[col].values
            
            # 5. 진입 판단 (히스테리시스 + TTL)
            print("진입 신호 체크 중...")
            decision = self.trading_engine.decide_on_bar_close(
                df_30m_with_features,
                len(df_30m_with_features) - 1
            )
            
            # 6. 진입 신호 있으면 로그 기록
            if decision['should_enter']:
                print(f"\n✅ 진입 신호 발생!")
                print(f"  방향: {decision['side']}")
                print(f"  확률: {decision['probability']:.4f}")
                print(f"  레짐: {decision['regime']}")
                
                # 거래 ID 생성
                trade_id = f"{self.symbol}_{decision['entry_ts'].strftime('%Y%m%d_%H%M%S')}"
                
                # 진입가/만기가 계산 (실제로는 API로 체결)
                entry_price = df_30m_with_features.iloc[-1]['close']  # 임시
                
                # 로그 기록 (결과는 나중에 업데이트)
                self.log_manager.log_trade(
                    trade_id=trade_id,
                    bar30_start=decision['bar30_start'],
                    bar30_end=decision['bar30_end'],
                    entry_ts=decision['entry_ts'],
                    label_ts=decision['label_ts'],
                    m1_index_entry=int(decision['entry_ts'].timestamp() // 60),
                    m1_index_label=int(decision['label_ts'].timestamp() // 60),
                    entry_price=entry_price,
                    label_price=0.0,  # 나중에 업데이트
                    payout=self.config.PAYOUT_RATIO,
                    result=0,  # 나중에 업데이트
                    side=decision['side'],
                    regime=decision['regime'],
                    is_weekend=0,  # TODO: 계산
                    regime_score=0.0,  # TODO: 계산
                    adx=0.0,  # TODO: 계산
                    di_plus=0.0,
                    di_minus=0.0,
                    p_at_entry=decision['probability'],
                    dp_at_entry=0.0,  # TODO: 계산
                    cut_on=self.config.CUT_ON,
                    cut_off=self.config.CUT_OFF,
                    cross_time=decision['bar30_end'],
                    ttl_used_sec=0.0,
                    ttl_valid=True,
                    refractory_window=self.config.REFRACTORY_MINUTES,
                    filters_applied='',
                    reason_code=decision['reason'],
                    blocked_reason=decision['blocked_reason'],
                    mode='LIVE'
                )
                
                # 성능 추적 업데이트 (실제 결과는 30분 후)
                # 여기서는 시뮬레이션으로 즉시 업데이트
                result = np.random.randint(0, 2)  # TODO: 실제 결과 대기
                status = self.model_trainer.update_performance(result)
                
                # 재학습 트리거 체크
                if status['need_retrain'] and not self.retrain_in_progress:
                    print(f"\n🔄 재학습 트리거 발동!")
                    self.trigger_retrain()
            
            else:
                if decision['blocked_reason']:
                    print(f"  차단: {decision['blocked_reason']}")
                else:
                    print(f"  대기 중... (사유: {decision['reason']})")
            
            # 처리 완료 표시
            self.last_process_time = bar30_end
            
            return True
            
        except Exception as e:
            print(f"\n❌ 처리 중 오류: {e}")
            traceback.print_exc()
            return False
    
    # ==========================================
    # 재학습
    # ==========================================
    
    def trigger_retrain(self):
        """재학습 트리거"""
        if self.retrain_in_progress:
            print("이미 재학습 진행 중...")
            return
        
        self.retrain_in_progress = True
        
        print(f"\n{'='*60}")
        print(f"재학습 시작")
        print(f"{'='*60}")
        
        try:
            # 1. 백업
            print("\n1. 현재 모델 백업...")
            self.version_manager.backup_current_model(reason="재학습 전 백업")
            
            # 2. 최신 데이터 준비
            print("\n2. 학습 데이터 준비...")
            
            if len(self.df_1m_buffer) < 3000:
                print("  ⚠️ 데이터 부족 - 재학습 스킵")
                self.retrain_in_progress = False
                return
            
            # 피처 생성
            features = self.feature_engineer.create_feature_pool(
                self.df_1m_buffer, 
                lookback_bars=100
            )
            
            # 라벨 생성
            df_30m = self.tf_manager.aggregate_1m_to_30m(self.df_1m_buffer)
            target = self.feature_engineer.create_target_30m(df_30m)
            
            valid = target.notna() & (target.index >= 100)
            X_new = features
            y_new = target[valid].reset_index(drop=True)
            
            min_len = min(len(X_new), len(y_new))
            X_new = X_new.iloc[:min_len]
            y_new = y_new.iloc[:min_len]
            
            print(f"  학습 데이터: {len(X_new):,}건")
            
            # 3. 재학습 전 검증
            print("\n3. 재학습 전 검증...")
            
            before_validation = self.validator.validate_before_retrain(
                X_new, y_new, 
                X_old=None,  # TODO: 이전 데이터 저장하여 비교
                y_old=None,
                model_trainer=self.model_trainer
            )
            
            if not before_validation['passed']:
                print("\n❌ 재학습 전 검증 실패 - 재학습 중단")
                self.retrain_in_progress = False
                return
            
            # 4. 새 모델 학습
            print("\n4. 새 모델 학습...")
            
            new_trainer = ModelTrainer(self.config)
            
            # 기존 피처 재사용
            if self.model_trainer.selected_features:
                new_trainer.selected_features = self.model_trainer.selected_features
            else:
                new_trainer.feature_selection_regime(X_new, y_new, regime_col='regime', top_k=30)
            
            new_metrics = new_trainer.train_ensemble_regime(
                X_new, y_new, 
                regime_col='regime', 
                test_size=0.2
            )
            
            # 5. 재학습 후 검증
            print("\n5. 재학습 후 검증...")
            
            test_split = int(len(X_new) * 0.8)
            X_test = X_new.iloc[test_split:]
            y_test = y_new.iloc[test_split:]
            
            after_validation = self.validator.validate_after_retrain(
                self.model_trainer,
                new_trainer,
                X_test,
                y_test
            )
            
            if not after_validation['passed']:
                print("\n⚠️ 재학습 후 검증 실패")
                print("  하지만 성능이 개선되었다면 배포 진행...")
                
                # 성능 개선 여부 확인
                if after_validation['improvement'] <= 0:
                    print("  ❌ 성능 개선 없음 - 재학습 중단")
                    self.retrain_in_progress = False
                    return
            
            # 6. 새 모델 배포
            print("\n6. 새 모델 배포...")
            
            # 버전 생성
            version_id = self.version_manager.create_version(
                new_trainer,
                new_metrics,
                description=f"재학습 - 승률 {self.model_trainer.perf_tracker.get_current_win_rate():.2%}",
                tags=['retrain', 'auto']
            )
            
            # 모델 저장
            new_trainer.save_model()
            
            # 현재 모델 교체
            self.model_trainer = new_trainer
            self.trading_engine.model = new_trainer
            
            # 성능 추적 리셋
            self.model_trainer.perf_tracker.reset()
            
            print(f"\n✓ 재학습 완료 - 버전: {version_id}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ 재학습 실패: {e}")
            traceback.print_exc()
        
        finally:
            self.retrain_in_progress = False
    
    # ==========================================
    # 실행 제어
    # ==========================================
    
    def start(self):
        """메인 루프 시작"""
        if not self.initialize():
            print("초기화 실패")
            return
        
        self.running = True
        
        print(f"\n{'='*60}")
        print(f"트레이딩 시작")
        print(f"{'='*60}\n")
        
        while self.running:
            try:
                # 1분마다 실행
                success = self.process_tick()
                
                if not success:
                    print("⚠️ 처리 실패 - 재시도")
                
                # 대기
                time.sleep(self.config.UPDATE_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                print("\n\n사용자 중단")
                break
            
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {e}")
                traceback.print_exc()
                
                # 잠시 대기 후 재시도
                time.sleep(10)
        
        self.stop()
    
    def stop(self):
        """시스템 종료"""
        print(f"\n{'='*60}")
        print(f"시스템 종료 중...")
        print(f"{'='*60}")
        
        self.running = False
        
        # 성능 추적 상태 저장
        self.model_trainer.save_performance_state()
        
        # 일일 요약 출력
        self.log_manager.print_daily_summary()
        
        print("\n✓ 시스템 종료 완료")


# ==========================================
# 메인 실행
# ==========================================
def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='30분봉 바이너리 옵션 트레이딩 시스템')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--backtest', action='store_true', help='백테스트 모드')
    
    args = parser.parse_args()
    
    if args.backtest:
        print("백테스트 모드는 trading_engine.py를 직접 실행하세요")
        return
    
    # 메인 트레이더 생성 및 실행
    trader = MainTrader(symbol=args.symbol)
    
    try:
        trader.start()
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        traceback.print_exc()
    finally:
        trader.stop()


if __name__ == "__main__":
    main()