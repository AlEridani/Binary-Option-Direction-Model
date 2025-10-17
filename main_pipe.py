# main_pipe.py - 메인 자동화 파이프라인 (재시작 없이 모델/필터 갱신)

import os
import time
import json
import threading
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import Config
from model_train import TimeSeriesModelTrainer, ModelOptimizer, FeatureEngineer
from data_merge import DataMerger, DataValidator
from real_trade import RealTimeTrader, BinanceAPIClient, TradingMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - main_pipe - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger("main_pipe")


class MainPipeline:
    def __init__(self):
        Config.create_directories()
        self.config = Config
        self.model_trainer = TimeSeriesModelTrainer(Config)
        self.model_optimizer = ModelOptimizer(Config)
        self.data_merger = DataMerger(Config)
        self.api_client = BinanceAPIClient()
        self.real_trader: RealTimeTrader | None = None
        self.monitor = TradingMonitor(Config)

        self.is_running = False
        self.trading_thread: threading.Thread | None = None
        self.last_retrain_time: datetime | None = None

        logger.info("메인 파이프라인 초기화 완료")

    # ---------- init ----------
    def initialize_system(self) -> bool:
        logger.info("시스템 초기화 시작")
        model_path = os.path.join(self.config.MODEL_DIR, "current_model.pkl")
        if not os.path.exists(model_path):
            logger.info("학습 모델 없음 → 초기 학습 플로우 시작")
            data = self.fetch_initial_training_data()
            if data is None or data.empty:
                logger.error("초기 데이터 수집 실패")
                return False
            metrics = self.model_optimizer.initial_training(data)
            logger.info(f"초기 학습 완료 (val wr={metrics['validation'].get('win_rate', 0):.2%})")
        else:
            if not self.model_trainer.load_model():
                logger.error("모델 로드 실패")
                return False
            logger.info("기존 모델 로드 완료")

        self.real_trader = RealTimeTrader(self.config, self.model_trainer, self.api_client)
        return True

    # ---------- data fetch ----------
    def fetch_initial_training_data(self) -> pd.DataFrame | None:
        logger.info("초기 학습 데이터 수집")
        try:
            # 시뮬레이션: 9개월 분(대용)
            return self.generate_simulation_data(days=270)
        except Exception as e:
            logger.error(f"초기 데이터 생성 오류: {e}")
            return None

    def generate_simulation_data(self, days=270):
        periods = days * 24 * 60
        ts = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq="1min", tz="UTC")
        np.random.seed(42)
        rets = np.random.normal(0.0001, 0.005, periods)
        price = 42000 * np.exp(np.cumsum(rets))
        rows = []
        for i, t in enumerate(ts):
            base = price[i]
            o = base * (1 + np.random.uniform(-0.001, 0.001))
            c = base * (1 + np.random.uniform(-0.001, 0.001))
            h = max(o, c) * (1 + np.random.uniform(0, 0.002))
            l = min(o, c) * (1 - np.random.uniform(0, 0.002))
            v = np.random.uniform(100, 1000) * (1 + 0.5 * np.sin(i / 1440 * 2 * np.pi))
            rows.append({"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        df = pd.DataFrame(rows)

        # 일자별 raw로 저장
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
        for d, g in df.groupby("date"):
            path = os.path.join(self.config.PRICE_DATA_DIR, "raw", f"prices_{pd.Timestamp(d).strftime('%Y%m%d')}.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                old = pd.read_csv(path)
                old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
                out = pd.concat([old, g.drop(columns=["date"])], ignore_index=True)
                out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                out.to_csv(path, index=False)
            else:
                g.drop(columns=["date"]).to_csv(path, index=False)
        return df.drop(columns=["date"])

    def fetch_live_data(self):
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
            if not df.empty:
                self.data_merger.add_new_price_data(df.tail(10))
                return df
        except Exception as e:
            logger.error(f"라이브 데이터 수집 오류: {e}")
        return None

    # ---------- retrain ----------
    def check_retrain_flag(self) -> bool:
        flag = os.path.join(self.config.BASE_DIR, ".retrain_required")
        if os.path.exists(flag):
            with open(flag, "r", encoding="utf-8") as f:
                _ = f.read()
            os.remove(flag)
            logger.info("재학습 플래그 감지")
            return True
        return False

    def retrain_pipeline(self) -> bool:
        logger.info("=" * 60)
        logger.info("재학습 파이프라인 시작")
        logger.info("=" * 60)
        try:
            logger.info("1) 데이터 병합")
            merged = self.data_merger.merge_all_data()
            if merged is None or merged.empty:
                logger.error("병합 데이터 없음")
                return False

            logger.info("2) 데이터 검증")
            ok, issues = DataValidator.validate_price_data(merged)
            if not ok:
                logger.warning(f"검증 이슈: {issues}")

            logger.info("3) 재학습")
            metrics = self.model_optimizer.retrain_model(merged)
            logger.info(f"재학습 완료: val acc={metrics['validation']['accuracy']:.4f}")

            logger.info("4) 실패패턴 분석→필터 저장")
            trades = self.data_merger.load_trade_logs()
            if not trades.empty:
                patterns = self.model_optimizer.analyze_failures(trades)
                if patterns:
                    out = os.path.join(self.config.FEATURE_LOG_DIR, "trade_filters.json")
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(patterns, f, indent=2)
                    logger.info(f"필터 업데이트 저장: {out}")

            # 5) 실시간 트레이더에 즉시 반영(재기동 없이)
            if self.real_trader:
                self.real_trader.model_trainer.load_model()
                self.real_trader.load_filters()
                self.real_trader.on_model_refreshed()
                logger.info("실시간 트레이더 새 모델/필터 적용 완료")

            self.last_retrain_time = datetime.now(timezone.utc)
            logger.info("재학습 파이프라인 완료")
            return True
        except Exception as e:
            logger.exception(f"재학습 파이프라인 오류: {e}")
            return False

    # ---------- loops ----------
    def trading_loop(self):
        logger.info("거래 루프 시작")
        while self.is_running:
            try:
                # 실시간 거래 1시간 세션
                self.real_trader.run_live_trading(duration_hours=1, entry_interval_minutes=1)
                # 리포트
                self.monitor.generate_report()
                # 재학습 체크
                if self.check_retrain_flag():
                    self.retrain_pipeline()
            except Exception as e:
                logger.exception(f"거래 루프 오류: {e}")
                time.sleep(60)

    # ---------- public ----------
    def start(self):
        logger.info("=" * 60)
        logger.info("시스템 시작")
        logger.info("=" * 60)
        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return
        self.is_running = True

        # 거래 스레드
        t = threading.Thread(target=self.trading_loop, daemon=True)
        t.start()
        self.trading_thread = t
        logger.info("모든 서비스 시작")

        # 메인 스레드: 상태 모니터
        try:
            while self.is_running:
                time.sleep(10)
                if self.trading_thread and not self.trading_thread.is_alive():
                    logger.warning("거래 스레드 중지 → 재시작")
                    self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
                    self.trading_thread.start()
        except KeyboardInterrupt:
            logger.info("종료 신호 수신")
            self.stop()

    def stop(self):
        logger.info("시스템 종료 중")
        self.is_running = False
        if self.real_trader:
            self.real_trader.is_running = False
        self.monitor.generate_report()
        if self.data_merger.merged_data is not None:
            self.data_merger.save_merged_data(self.data_merger.merged_data)
        logger.info("시스템 종료 완료")

    def run_backtest(self, start_date=None, end_date=None):
        logger.info("백테스팅 모드")
        if not self.initialize_system():
            logger.error("초기화 실패")
            return
        hist = self.data_merger.load_price_data(start_date, end_date)
        if hist.empty:
            logger.error("백테스트 데이터 없음")
            return
        # 실제 백테스트는 RealTimeTrader.backtest를 생략(요구 스펙상 실시간 구조 우선)

    def optimize_hyperparameters(self):
        logger.info("하이퍼파라미터 최적화(샘플, 길이 문제로 생략 가능)")
        X, y = self.data_merger.get_training_data(lookback_days=90)
        if X is None:
            logger.error("학습 데이터 로드 실패")
            return
        # 필요시 Grid/DiffSearch 구현 가능 (현 버전은 생략)


ModelTrainer = TimeSeriesModelTrainer

if __name__ == "__main__":
    MainPipeline().start()
