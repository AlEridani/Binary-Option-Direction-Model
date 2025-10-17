# main_pipe.py - 메인 자동화 파이프라인 (real_trade 최신 구조 대응 + 재학습 중 신규진입 일시정지)
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import schedule

from config import Config
from model_train import TimeSeriesModelTrainer, ModelOptimizer, FeatureEngineer
from data_merge import DataMerger, DataValidator
from real_trade import RealTimeTrader, BinanceAPIClient, TradingMonitor

# 로깅 설정
logger = logging.getLogger("main_pipe")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    # 파일 핸들러
    fh = logging.FileHandler('pipeline.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

# 외부 호환성(다른 모듈에서 ModelTrainer 이름만 import 하더라도 OK)
ModelTrainer = TimeSeriesModelTrainer


class MainPipeline:
    """메인 자동화 파이프라인"""

    def __init__(self):
        # 디렉토리 생성
        Config.create_directories()
        self.config = Config

        # 컴포넌트
        self.model_trainer = ModelTrainer(Config)
        self.model_optimizer = ModelOptimizer(Config)
        self.data_merger = DataMerger(Config)
        self.api_client = BinanceAPIClient()
        self.real_trader: RealTimeTrader | None = None
        self.monitor = TradingMonitor(Config)

        # 상태
        self.is_running = False
        self.last_retrain_time = None
        self.trading_thread = None

        logger.info("메인 파이프라인 초기화 완료")

    # ---------------- 초기화/데이터 ----------------
    def initialize_system(self) -> bool:
        logger.info("시스템 초기화 시작...")

        model_path = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        if not os.path.exists(model_path):
            logger.info("학습된 모델이 없습니다. 초기 학습을 시작합니다.")
            initial_data = self.fetch_initial_training_data()
            if initial_data is None or initial_data.empty:
                logger.error("초기 학습용 데이터를 가져올 수 없습니다.")
                return False

            metrics = self.model_optimizer.initial_training(initial_data)
            logger.info(f"초기 학습 완료. 테스트 승률: {metrics['test']['win_rate']:.2%}")
        else:
            if self.model_trainer.load_model():
                logger.info("기존 모델 로드 완료")
            else:
                logger.error("모델 로드 실패")
                return False

        # 실시간 거래 객체
        self.real_trader = RealTimeTrader(self.config, self.model_trainer, self.api_client)
        return True

    def fetch_initial_training_data(self) -> pd.DataFrame | None:
        logger.info("초기 학습 데이터 수집 중...")
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
            if df is not None and not df.empty:
                self.save_price_data(df)
                logger.info(f"초기 데이터 수집 완료: {len(df)} 레코드")
                return df
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")

        logger.info("시뮬레이션 데이터 생성으로 대체")
        return self.generate_simulation_data(days=270)  # 9개월

    def generate_simulation_data(self, days=270) -> pd.DataFrame:
        periods = days * 24 * 60  # 1분봉
        ts = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1min', tz='UTC')
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, periods)
        price = 42000 * np.exp(np.cumsum(returns))

        rows = []
        for i, t in enumerate(ts):
            base = price[i]
            o = base * (1 + np.random.uniform(-0.001, 0.001))
            c = base * (1 + np.random.uniform(-0.001, 0.001))
            h = max(o, c) * (1 + np.random.uniform(0, 0.002))
            l = min(o, c) * (1 - np.random.uniform(0, 0.002))
            v = np.random.uniform(100, 1000) * (1 + 0.5 * np.sin(i/1440 * 2 * np.pi))
            rows.append({'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

        df = pd.DataFrame(rows)
        self.save_price_data(df)
        return df

    def save_price_data(self, df: pd.DataFrame):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['date'] = df['timestamp'].dt.date
        for date, group in df.groupby('date'):
            filename = f"prices_{pd.Timestamp(date).strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.config.PRICE_DATA_DIR, 'raw', filename)
            if os.path.exists(filepath):
                existing = pd.read_csv(filepath)
                existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True, errors='coerce')
                combined = pd.concat([existing, group.drop(columns=['date'])], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                combined.to_csv(filepath, index=False)
            else:
                group.drop(columns=['date']).to_csv(filepath, index=False)

    def fetch_live_data(self) -> pd.DataFrame | None:
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
            if df is not None and not df.empty:
                self.data_merger.add_new_price_data(df.tail(10))  # 최근 10개만 추가
                return df
        except Exception as e:
            logger.error(f"실시간 데이터 수집 오류: {e}")
        return None

    # ---------------- 재학습 ----------------
    def check_retrain_flag(self) -> bool:
        flag_path = os.path.join(self.config.BASE_DIR, '.retrain_required')
        if os.path.exists(flag_path):
            with open(flag_path, 'r') as f:
                flag_time = f.read().strip()
            os.remove(flag_path)
            logger.info(f"재학습 플래그 감지: {flag_time}")
            return True
        return False

    def retrain_pipeline(self) -> bool:
        logger.info("="*60)
        logger.info("재학습 파이프라인 시작")
        logger.info("="*60)
        try:
            # ★ 재학습 도중 신규 진입 일시정지
            if self.real_trader:
                self.real_trader.pause_new_entries = True

            # 1) 병합
            logger.info("1. 데이터 병합 중...")
            merged = self.data_merger.merge_all_data()
            if merged is None or merged.empty:
                logger.error("병합할 데이터가 없습니다.")
                return False

            # 2) 검증
            logger.info("2. 데이터 검증 중...")
            is_valid, issues = DataValidator.validate_price_data(merged)
            if not is_valid:
                logger.warning(f"데이터 문제: {issues}")

            # 3) 재학습
            logger.info("3. 모델 재학습 중...")
            metrics = self.model_optimizer.retrain_model(merged)
            logger.info(f"재학습 완료:"
                        f" train_acc={metrics.get('train', {}).get('accuracy', 0):.4f}, "
                        f" val_acc={metrics.get('validation', {}).get('accuracy', 0):.4f}, "
                        f" test_acc={metrics.get('test', {}).get('accuracy', 0):.4f}")

            # 4) 실패 분석 -> 필터 저장
            logger.info("4. 거래 실패 분석 중...")
            trade_logs = self.data_merger.load_trade_logs()
            if trade_logs is not None and not trade_logs.empty:
                failure_patterns = self.model_optimizer.analyze_failures(trade_logs)
                if failure_patterns:
                    filter_path = os.path.join(self.config.FEATURE_LOG_DIR, 'trade_filters.json')
                    with open(filter_path, 'w') as f:
                        json.dump(failure_patterns, f, indent=2)
                    logger.info(f"거래 필터 업데이트: {list(failure_patterns.keys())}")

            # 5) 실시간 모듈에 새 모델 반영 + 승률/평가히스토리 초기화
            if self.real_trader:
                self.real_trader.model_trainer = self.model_optimizer.trainer
                self.real_trader.model_trainer.load_model()   # 안전하게 디스크에서 재로드
                self.real_trader.load_filters()
                self.real_trader.reset_winrate("model_retrained")
                logger.info("실시간 모듈에 새 모델 적용 및 승률/평가히스토리 초기화 완료")

            self.last_retrain_time = datetime.utcnow()
            logger.info("재학습 파이프라인 완료")
            return True

        except Exception as e:
            logger.exception(f"재학습 파이프라인 오류: {e}")
            return False
        finally:
            # 신규 진입 재개
            if self.real_trader:
                self.real_trader.pause_new_entries = False

    # ---------------- 거래 루프/스케줄 ----------------
    def trading_loop(self):
        logger.info("거래 루프 시작")
        while self.is_running:
            try:
                # 실시간 거래 실행 (1시간 세션 단위)
                self.real_trader.run_live_trading(duration_hours=1, amount=100.0)

                # 세션 종료 후 리포트
                self.monitor.generate_report()

                # 플래그 기반 재학습
                if self.check_retrain_flag():
                    self.retrain_pipeline()

            except Exception as e:
                logger.error(f"거래 루프 오류: {e}")
                time.sleep(60)  # 오류 시 잠시 대기 후 재시도

    def scheduled_tasks(self):
        # 매일 자정: 데이터 정리
        schedule.every().day.at("00:00").do(self.daily_maintenance)
        # 4시간마다: 리포트
        schedule.every(4).hours.do(self.monitor.generate_report)
        # 12시간마다: 자동 재학습 검토
        schedule.every(12).hours.do(self.auto_retrain_check)

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)

    def daily_maintenance(self):
        logger.info("일일 유지보수 시작")
        # 오래된 데이터 정리
        self.data_merger.cleanup_old_data(days_to_keep=90)
        # 로그 로테이션
        log_file = 'pipeline.log'
        if os.path.exists(log_file):
            size_mb = os.path.getsize(log_file) / (1024 * 1024)
            if size_mb > 100:
                archive = f"pipeline_{datetime.utcnow().strftime('%Y%m%d')}.log"
                os.rename(log_file, archive)
                logger.info(f"로그 아카이브: {archive}")
        # 30일 통계 저장
        stats = self.monitor.analyze_recent_trades(30)
        if stats:
            stats_path = os.path.join(self.config.BASE_DIR, f"stats_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        logger.info("일일 유지보수 완료")

    def auto_retrain_check(self):
        logger.info("자동 재학습 검토 중...")
        stats = self.monitor.analyze_recent_trades(1)  # 최근 1일
        if stats and stats['total_trades'] >= 20:
            wr = stats.get('win_rate', 0)
            if wr < self.config.RETRAIN_THRESHOLD:
                logger.info(f"승률 {wr:.2%} < {self.config.RETRAIN_THRESHOLD:.0%} → 자동 재학습 시작")
                self.retrain_pipeline()
            else:
                logger.info(f"현재 승률 {wr:.2%} → 재학습 불필요")
        else:
            logger.info("거래 데이터 부족 → 재학습 검토 스킵")

    # ---------------- 구동/정지 ----------------
    def start(self):
        logger.info("="*60)
        logger.info("바이너리 옵션 예측 시스템 시작")
        logger.info("="*60)

        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        self.is_running = True

        # 거래 스레드
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

        # 스케줄 스레드
        schedule_thread = threading.Thread(target=self.scheduled_tasks, daemon=True)
        schedule_thread.start()

        logger.info("모든 서비스 시작 완료")

        # 메인 루프: 상태 모니터링
        try:
            while self.is_running:
                time.sleep(10)
                self.monitor_system_health()
        except KeyboardInterrupt:
            logger.info("시스템 종료 신호 수신")
            self.stop()

    def monitor_system_health(self):
        if self.trading_thread and not self.trading_thread.is_alive():
            logger.warning("거래 스레드가 중지됨. 재시작...")
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()

    def stop(self):
        logger.info("시스템 종료 중...")
        self.is_running = False
        if self.real_trader:
            self.real_trader.is_running = False
            self.real_trader.print_performance_summary()
        self.monitor.generate_report()
        if getattr(self.data_merger, "merged_data", None) is not None:
            self.data_merger.save_merged_data()
        logger.info("시스템 종료 완료")

    # ---------------- 백테스트 ----------------
    def run_backtest(self, start_date: datetime | None = None, end_date: datetime | None = None):
        """
        메인파이프 내부 백테스트:
        - FeatureEngineer로 피처/타깃 생성
        - 과거 순서 유지, 미래데이터 누출 방지
        - 고정 베팅(금액 100, 승리시 WIN_RATE 수익, 패배시 -100)
        """
        logger.info("백테스팅 모드 시작")

        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        # 가격 데이터 로드
        historical = self.data_merger.load_price_data(start_date, end_date)
        if historical is None or historical.empty:
            logger.error("백테스팅용 데이터가 없습니다.")
            return

        # 피처/타깃
        fe = FeatureEngineer()
        features = fe.create_feature_pool(historical, lookback_window=200)
        target = fe.create_target(historical, window=self.config.PREDICTION_WINDOW)

        valid = target.notna()
        features = features[valid]
        target = target[valid]
        tail = self.config.PREDICTION_WINDOW
        if len(features) > tail:
            features = features.iloc[:-tail]
            target = target.iloc[:-tail]

        if not self.model_trainer.models:
            if not self.model_trainer.load_model():
                logger.error("모델을 찾을 수 없습니다. 먼저 학습을 진행하세요.")
                return

        amount = 100.0
        preds_proba = self.model_trainer.predict_proba(features)
        preds = (preds_proba > 0.5).astype(int)

        correct = (preds == target.values)
        wins = int(correct.sum())
        losses = int((~correct).sum())
        total = int(len(correct))
        win_rate = wins / total if total > 0 else 0.0
        pnl = wins * amount * self.config.WIN_RATE - losses * amount

        logger.info(f"백테스트 결과: 기간={features.index.min()}~{features.index.max()} "
                    f"총거래={total} 승={wins} 패={losses} 승률={win_rate:.2%} PnL=${pnl:.2f}")

        out = pd.DataFrame({
            'timestamp': historical.loc[features.index, 'timestamp'].values if 'timestamp' in historical.columns else features.index,
            'pred': preds,
            'actual': target.values,
            'correct': correct.astype(int),
            'proba': preds_proba
        })
        out.to_csv('backtest_results.csv', index=False)
        logger.info("백테스팅 결과 저장: backtest_results.csv")


# ---------------- CLI ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='바이너리 옵션 예측 시스템')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'optimize'],
                        default='live', help='실행 모드')
    parser.add_argument('--start-date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    args = parser.parse_args()

    pipeline = MainPipeline()

    # 커스텀 설정 적용(Optional)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom = json.load(f)
            for k, v in custom.items():
                if hasattr(Config, k):
                    setattr(Config, k, v)

    if args.mode == 'live':
        pipeline.start()

    elif args.mode == 'backtest':
        sd = pd.to_datetime(args.start_date) if args.start_date else None
        ed = pd.to_datetime(args.end_date) if args.end_date else None
        pipeline.run_backtest(sd, ed)

    elif args.mode == 'train':
        if not pipeline.initialize_system():
            logger.error("시스템 초기화 실패")
            return
        data = pipeline.fetch_initial_training_data()
        if data is not None and not data.empty:
            metrics = pipeline.model_optimizer.initial_training(data)
            logger.info("학습 완료")
            for split, metric in metrics.items():
                logger.info(f"{split}: {metric}")

    elif args.mode == 'optimize':
        logger.info("하이퍼파라미터 최적화: 별도 구현을 권장합니다.")
        # 필요 시 pipeline.data_merger.get_training_data(lookback_days=90) 등 활용

if __name__ == "__main__":
    main()
