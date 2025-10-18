# main_pipe.py - 메인 자동화 파이프라인 (무결성검사 + A/B 교체 + 실시간 반영)

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config import Config
from model_train import TimeSeriesModelTrainer, ModelOptimizer  # 내부에서 FeatureEngineer 사용
from data_merge import DataMerger, DataValidator
from real_trade import RealTimeTrader, BinanceAPIClient
from trading_monitor import TradingMonitor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger("main_pipe")


class MainPipeline:
    """메인 자동화 파이프라인"""

    def __init__(self):
        # 디렉토리 준비
        Config.create_directories()
        self.config = Config

        # 코어 컴포넌트
        self.model_trainer = TimeSeriesModelTrainer(self.config)
        self.model_optimizer = ModelOptimizer(self.config)   # 내부에서 FeatureEngineer로 피처 생성+선택
        self.data_merger = DataMerger(self.config)
        self.api_client = BinanceAPIClient()
        self.real_trader: RealTimeTrader | None = None
        self.monitor = TradingMonitor(self.config)

        # 상태
        self.is_running = False
        self.last_retrain_time: datetime | None = None
        self.trading_thread: threading.Thread | None = None
        self.scheduler_thread: threading.Thread | None = None

        logger.info("메인 파이프라인 초기화 완료")

    # -----------------------------
    # 초기화 & 데이터 수집
    # -----------------------------
    def initialize_system(self) -> bool:
        """모델 존재하면 로드, 없으면 초기 학습 진행 후 트레이더 준비"""
        logger.info("시스템 초기화 시작...")
        model_path = os.path.join(self.config.MODEL_DIR, "current_model.pkl")

        if not os.path.exists(model_path):
            logger.info("학습된 모델 없음 → 초기 학습 진행")
            initial_data = self.fetch_initial_training_data()
            if initial_data is None or initial_data.empty:
                logger.error("초기 학습용 데이터를 가져올 수 없습니다.")
                return False

            metrics = self.model_optimizer.initial_training(initial_data)
            logger.info(
                "초기 학습 완료 | train: {t:.2%}, val: {v:.2%}, test: {s:.2%}".format(
                    t=metrics["train"]["win_rate"],
                    v=metrics["validation"]["win_rate"],
                    s=metrics["test"]["win_rate"],
                )
            )
        else:
            if not self.model_trainer.load_model(model_path):
                logger.error("기존 모델 로드 실패")
                return False
            logger.info("기존 모델 로드 완료")

        # 실시간 트레이더 준비
        self.real_trader = RealTimeTrader(self.config, self.model_trainer, self.api_client)
        logger.info("실시간 거래 모듈 초기화 완료")
        return True

    def fetch_initial_training_data(self) -> pd.DataFrame | None:
        """초기 학습용 데이터: API 시도 → 실패 시 시뮬레이션"""
        logger.info("초기 학습 데이터 수집 중...")
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=1500)
            if df is not None and not df.empty:
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                self.save_price_data(df)
                logger.info(f"초기 데이터 수집 완료: {len(df)} rows")
                return df
        except Exception as e:
            logger.warning(f"API 수집 실패(시뮬레이션 대체): {e}")

        # 시뮬레이션 (약 9개월)
        sim = self.generate_simulation_data(days=270)
        logger.info(f"시뮬레이션 데이터 생성 완료: {len(sim)} rows")
        return sim

    def generate_simulation_data(self, days=270) -> pd.DataFrame:
        """현실적인 1분봉 시뮬레이션"""
        periods = days * 24 * 60
        ts = pd.date_range(end=datetime.utcnow(), periods=periods, freq="1min")
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
            v = np.random.uniform(100, 1000) * (1 + 0.5 * np.sin(i / 1440 * 2 * np.pi))
            rows.append(
                {"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": v}
            )

        df = pd.DataFrame(rows)
        self.save_price_data(df)
        return df

    def save_price_data(self, df: pd.DataFrame) -> None:
        """raw 가격데이터 일자별 저장(중복 제거)"""
        if "timestamp" not in df.columns:
            return
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        for d, g in df.groupby("date"):
            fname = f"prices_{pd.Timestamp(d).strftime('%Y%m%d')}.csv"
            fpath = os.path.join(self.config.PRICE_DATA_DIR, "raw", fname)

            g = g.drop(columns=["date"])
            if os.path.exists(fpath):
                exist = pd.read_csv(fpath)
                exist["timestamp"] = pd.to_datetime(exist["timestamp"])
                merged = (
                    pd.concat([exist, g], ignore_index=True)
                    .drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                )
                merged.to_csv(fpath, index=False)
            else:
                g.to_csv(fpath, index=False)

    # -----------------------------
    # 재학습 파이프라인 (무결성검사 + A/B 비교 + 실시간 반영)
    # -----------------------------
    def retrain_pipeline(self) -> bool:
        logger.info("=" * 60)
        logger.info("재학습 파이프라인 시작")
        logger.info("=" * 60)

        try:
            # (A) 재학습 전 무결성 검사
            if not self.monitor.validate_data_integrity():
                logger.error("무결성 검사 실패 → 재학습 중단")
                return False

            # 1. 데이터 병합
            logger.info("1. 데이터 병합 중...")
            merged = self.data_merger.merge_all_data()
            if merged is None or merged.empty:
                logger.error("병합할 데이터가 없습니다.")
                return False

            # 2. 새 모델 학습(임시) — 내부에서 FeatureEngineer로 피처 생성/선택
            logger.info("2. 모델 재학습(새 모델) 진행...")
            _new_metrics = self.model_optimizer.retrain_model(merged)
            new_trainer = self.model_optimizer.trainer  # 새 모델 핸들러

            # 3. A/B 비교 후 교체/롤백 (tail 데이터 기준)
            tmp = TimeSeriesModelTrainer(self.config)
            X, y = tmp._prepare_xy(merged)

            old = TimeSeriesModelTrainer(self.config)
            old.load_model()

            applied, m_old, m_new = self.monitor.compare_and_apply(old, new_trainer, X, y)

            # 4. 실시간 트레이더에 즉시 반영 + 필터 리로드 + 승률 초기화(교체 시)
            if self.real_trader:
                self.real_trader.model_trainer.load_model()
                self.real_trader.load_filters()
                if applied and hasattr(self.real_trader, "reset_performance"):
                    self.real_trader.reset_performance()
                logger.info("실시간 거래 모듈에 모델/필터 적용 완료")

            self.last_retrain_time = datetime.now()
            logger.info("재학습 파이프라인 완료")
            return True

        except Exception as e:
            logger.error(f"재학습 파이프라인 오류: {e}")
            return False

    # -----------------------------
    # 루프/스케줄
    # -----------------------------
    def trading_loop(self):
        """실시간 거래 루프(1시간 단위 실행 반복)"""
        logger.info("거래 루프 시작")
        while self.is_running:
            try:
                if self.real_trader is None:
                    logger.warning("RealTimeTrader가 준비되지 않음. 초기화 재시도...")
                    ok = self.initialize_system()
                    if not ok:
                        time.sleep(10)
                        continue

                # 1시간 실행 (RealTimeTrader 내부에서 1분 신호/동시보유/10분 만기 관리)
                self.real_trader.run_live_trading(duration_hours=1)

                # 루프 사이 휴지
                time.sleep(2)

            except Exception as e:
                logger.error(f"거래 루프 오류: {e}")
                time.sleep(30)

    def scheduled_tasks(self):
        """간단한 주기 작업: 일일 유지보수 + 자동 재학습 체크"""
        logger.info("스케줄러 쓰레드 시작")
        while self.is_running:
            try:
                now = datetime.now()
                # 매 정시 + 5분에 간단 유지보수
                if now.minute == 5 and now.second < 3:
                    self.daily_maintenance()
                    time.sleep(3)

                # 50건 기준 자동 재학습 플래그 파일 감지(RealTimeTrader가 생성 가능)
                flag_path = os.path.join(self.config.BASE_DIR, ".retrain_required")
                if os.path.exists(flag_path):
                    with open(flag_path, "r") as f:
                        ts = f.read().strip()
                    os.remove(flag_path)
                    logger.info(f"재학습 플래그 감지: {ts}")
                    self.retrain_pipeline()

                time.sleep(1)
            except Exception as e:
                logger.error(f"스케줄러 오류: {e}")
                time.sleep(5)

    # -----------------------------
    # 유지보수/백테스트
    # -----------------------------
    def daily_maintenance(self):
        """경량 유지보수: 오래된 파일 정리 + 간단 리포트 저장"""
        logger.info("일일 유지보수 시작")
        # 가격/거래 로그 정리
        self.data_merger.cleanup_old_data(days_to_keep=90)

        # 간단 성능 스냅샷
        stats = self.monitor.analyze_recent_trades(7)
        if stats:
            p = os.path.join(self.config.BASE_DIR, f"stats_{datetime.now().strftime('%Y%m%d')}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"주간 성능 스냅샷 저장: {p}")

        logger.info("일일 유지보수 완료")

    def run_backtest(self, start_date: datetime | None = None, end_date: datetime | None = None):
        """백테스팅: 병합 데이터 기반으로 RealTimeTrader의 backtest 사용"""
        logger.info("백테스팅 모드 시작")
        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        historical = self.data_merger.load_price_data(start_date, end_date)
        if historical.empty:
            logger.error("백테스팅용 데이터가 없습니다.")
            return

        # 트레이더 백테스트 호출
        results = self.real_trader.backtest(historical, start_date, end_date)
        if results is not None:
            results.to_csv("backtest_results.csv", index=False)
            logger.info("백테스트 결과 저장: backtest_results.csv")

    # -----------------------------
    # 시작/중지
    # -----------------------------
    def start(self):
        logger.info("=" * 60)
        logger.info("바이너리 옵션 예측 시스템 시작")
        logger.info("=" * 60)

        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        self.is_running = True

        # 거래 쓰레드
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

        # 스케줄 쓰레드
        self.scheduler_thread = threading.Thread(target=self.scheduled_tasks, daemon=True)
        self.scheduler_thread.start()

        logger.info("모든 서비스 시작 완료")

        try:
            while self.is_running:
                time.sleep(2)
                # 스레드 헬스체크
                if self.trading_thread and not self.trading_thread.is_alive():
                    logger.warning("거래 스레드 중지 감지 → 재시작")
                    self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
                    self.trading_thread.start()
        except KeyboardInterrupt:
            logger.info("시스템 종료 신호 수신")
            self.stop()

    def stop(self):
        logger.info("시스템 종료 중...")
        self.is_running = False

        if self.real_trader:
            self.real_trader.is_running = False
            if hasattr(self.real_trader, "print_performance_summary"):
                self.real_trader.print_performance_summary()

        self.monitor.generate_report()
        if self.data_merger.merged_data is not None:
            self.data_merger.save_merged_data()

        logger.info("시스템 종료 완료")


# CLI 엔트리 (직접 실행 시)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="바이너리 옵션 예측 시스템")
    parser.add_argument("--mode", choices=["live", "backtest", "train", "retrain"], default="live")
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    args = parser.parse_args()

    pipe = MainPipeline()

    if args.mode == "live":
        pipe.start()
    elif args.mode == "backtest":
        sd = pd.to_datetime(args.start_date) if args.start_date else None
        ed = pd.to_datetime(args.end_date) if args.end_date else None
        pipe.run_backtest(sd, ed)
    elif args.mode == "train":
        pipe.initialize_system()  # 초기 학습은 initialize_system에서 수행됨(없을 때)
    elif args.mode == "retrain":
        pipe.initialize_system()
        pipe.retrain_pipeline()
