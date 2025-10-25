# main_pipe.py
# 메인 자동화 파이프라인 (논블로킹 재학습/적응형 필터 호환)

import os
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

from config import Config
from model_train import ModelTrainer, ModelOptimizer
from feature_engineer import FeatureEngineer
from data_merge import DataMerger, DataValidator
from real_trade import RealTimeTrader, BinanceAPIClient
from monitor import Monitor

# ──────────────────────────────────────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_pipe")


# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _coerce_to_utc(series: pd.Series) -> pd.Series:
    """Naive/문자열/혼합을 모두 UTC-aware로 통일"""
    s = pd.to_datetime(series, errors='coerce', utc=False)
    try:
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
    except Exception:
        def _one(x):
            if pd.isna(x):
                return pd.NaT
            dt = pd.to_datetime(x, errors='coerce')
            if pd.isna(dt):
                return pd.NaT
            if getattr(dt, "tzinfo", None) is None:
                return dt.tz_localize("UTC")
            return dt.tz_convert("UTC")
        s = series.apply(_one).astype("datetime64[ns, UTC]")
    return s


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + (z * z) / n
    center = p + (z * z) / (2 * n)
    margin = z * ((p * (1 - p) + (z * z) / (4 * n)) / n) ** 0.5
    return (center - margin) / denom


def _build_adaptive_filters(failure_patterns: dict) -> dict:
    """
    ModelOptimizer.analyze_failures() 출력 → RealTimeTrader.load_adaptive_filters() 구조 변환
    """
    active = []
    if not failure_patterns:
        return {"active_filters": [], "filter_history": []}

    # volatility
    if "high_volatility" in failure_patterns:
        fp = failure_patterns["high_volatility"]
        th = float(fp.get("atr_14_threshold", 0.0))
        active.append({
            "name": "High Volatility Filter",
            "type": "risk",
            "field": "atr_14",
            "operator": ">",
            "threshold": th,
            "condition": fp.get("filter", f"atr_14 > {th}"),
            "reason": fp.get("reason", "고변동성 구간 실패율↑"),
            "improvement": 0.0
        })
    # volume
    if "high_volume" in failure_patterns:
        fp = failure_patterns["high_volume"]
        th = float(fp.get("volume_ratio_threshold", 1.5))
        active.append({
            "name": "High Volume Filter",
            "type": "volume",
            "field": "volume_ratio",
            "operator": ">",
            "threshold": th,
            "condition": fp.get("filter", f"volume_ratio > {th}"),
            "reason": fp.get("reason", "고거래량 구간 실패율↑"),
            "improvement": 0.0
        })
    if "low_volume" in failure_patterns:
        fp = failure_patterns["low_volume"]
        th = float(fp.get("volume_ratio_threshold", 0.5))
        active.append({
            "name": "Low Volume Filter",
            "type": "volume",
            "field": "volume_ratio",
            "operator": "<",
            "threshold": th,
            "condition": fp.get("filter", f"volume_ratio < {th}"),
            "reason": fp.get("reason", "저유동성 구간 실패율↑"),
            "improvement": 0.0
        })
    # time
    if "time_based" in failure_patterns:
        fp = failure_patterns["time_based"]
        bad_hours = list(map(int, fp.get("avoid_hours", [])))
        active.append({
            "name": "Bad Hours Filter",
            "type": "time",
            "field": "hour",
            "operator": "in",
            "bad_hours": bad_hours,
            "condition": fp.get("filter", f"hour in {bad_hours}"),
            "reason": fp.get("reason", "저승률 시간대"),
            "improvement": 0.0
        })
    # rsi
    if "rsi_extreme" in failure_patterns:
        fp = failure_patterns["rsi_extreme"]
        active.append({
            "name": "RSI Extreme Filter",
            "type": "rsi",
            "field": "rsi_14",
            "operator": "extreme",
            "lower_threshold": 30.0,
            "upper_threshold": 70.0,
            "condition": fp.get("condition", "RSI < 30 or RSI > 70"),
            "reason": fp.get("reason", "RSI 극단값 실패율↑"),
            "improvement": 0.0
        })

    return {"active_filters": active, "filter_history": []}


# ──────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────────────────────
class MainPipeline:
    """메인 자동화 파이프라인"""

    def __init__(self):
        # 디렉토리 준비
        Config.create_directories()
        self.config = Config

        # 컴포넌트
        self.model_trainer = ModelTrainer(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.data_merger = DataMerger(self.config)
        self.api_client = BinanceAPIClient()
        self.real_trader: Optional[RealTimeTrader] = None
        self.monitor = Monitor(self.config)  # 최근 성과 분석/리포트 담당

        # 상태
        self.is_running = False
        self.last_retrain_time: Optional[datetime] = None
        self.trading_thread: Optional[threading.Thread] = None

        logger.info("MainPipeline 초기화 완료")

    # ── 초기화 ────────────────────────────────────────────────────────────────
    def initialize_system(self) -> bool:
        logger.info("시스템 초기화 시작…")

        # 1) 모델 준비
        model_file = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        if not os.path.exists(model_file):
            logger.info("학습된 모델 없음 → 초기 학습 시작")
            initial_data = self.fetch_initial_training_data()
            if initial_data is not None and not initial_data.empty:
                metrics = self.model_optimizer.initial_training(initial_data)
                try:
                    logger.info(
                        "초기 학습 완료 | "
                        f"train={metrics.get('train')} | "
                        f"val={metrics.get('validation')} | "
                        f"test={metrics.get('test')}"
                    )
                except Exception:
                    logger.info("초기 학습 완료")
            else:
                logger.error("초기 학습 데이터 부족")
                return False
        else:
            if self.model_trainer.load_model():
                logger.info("기존 모델 로드 완료")
            else:
                logger.error("모델 로드 실패")
                return False

        # 2) 실시간 트레이더
        self.real_trader = RealTimeTrader(
            config=self.config,
            model_trainer=self.model_trainer,
            api_client=self.api_client
        )
        # 적응형 필터 프리로드(있으면)
        if hasattr(self.real_trader, "load_adaptive_filters"):
            try:
                self.real_trader.adaptive_filters = self.real_trader.load_adaptive_filters()
            except Exception:
                pass

        return True

    # ── 데이터 소스 ───────────────────────────────────────────────────────────
    def fetch_initial_training_data(self) -> Optional[pd.DataFrame]:
        logger.info("초기 학습 데이터 수집…")
        try:
            dfs = []
            # 실제에선 여러 구간/요청으로 확장
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=1000)
            if df is not None and not df.empty:
                dfs.append(df)
            if dfs:
                out = pd.concat(dfs, ignore_index=True)
                out = out.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                self.save_price_data(out)
                logger.info(f"초기 데이터 수집 완료: {len(out):,} rows")
                return out
        except Exception as e:
            logger.error(f"초기 데이터 수집 오류: {e}")

        # 실패 시 시뮬레이션
        logger.info("시뮬레이션 데이터 생성으로 대체")
        return self.generate_simulation_data(days=180)

    def generate_simulation_data(self, days=180) -> pd.DataFrame:
        periods = days * 24 * 60
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='1min')

        rng = np.random.default_rng(42)
        rets = rng.normal(0.00006, 0.004, periods)
        price = 42000 * np.exp(np.cumsum(rets))

        rows = []
        for i, ts in enumerate(idx):
            base = price[i]
            o = base * (1 + rng.uniform(-0.001, 0.001))
            c = base * (1 + rng.uniform(-0.001, 0.001))
            h = max(o, c) * (1 + rng.uniform(0, 0.002))
            l = min(o, c) * (1 - rng.uniform(0, 0.002))
            v = rng.uniform(100, 1200) * (1 + 0.3 * np.sin(i/1440 * 2*np.pi))
            rows.append({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

        df = pd.DataFrame(rows)
        self.save_price_data(df)
        return df

    def save_price_data(self, df: pd.DataFrame):
        os.makedirs(self.config.PRICE_DATA_DIR, exist_ok=True)
        path = os.path.join(self.config.PRICE_DATA_DIR, 'prices.csv')

        if 'timestamp' not in df.columns:
            raise ValueError("가격 DF에 'timestamp' 없음")
        df = df.copy()
        df['timestamp'] = _coerce_to_utc(df['timestamp'])
        df['ts_min'] = df['timestamp'].dt.floor('T')

        if os.path.exists(path):
            old = pd.read_csv(path)
            if 'timestamp' in old.columns:
                old['timestamp'] = _coerce_to_utc(old['timestamp'])
                if 'ts_min' not in old.columns:
                    old['ts_min'] = old['timestamp'].dt.floor('T')
            combined = pd.concat([old, df], ignore_index=True)
        else:
            combined = df

        combined = (
            combined.sort_values('timestamp')
                    .drop_duplicates(subset=['ts_min'], keep='last')
        )
        combined.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"가격 데이터 저장: {path} ({len(combined):,} rows)")

    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
            if df is not None and not df.empty:
                self.data_merger.add_new_price_data(df.tail(10))
                return df
        except Exception as e:
            logger.error(f"실시간 데이터 수집 오류: {e}")
        return None

    # ── 재학습 ────────────────────────────────────────────────────────────────
    def check_retrain_flag(self) -> bool:
        flag_path = os.path.join(self.config.BASE_DIR, '.retrain_required')
        if os.path.exists(flag_path):
            try:
                with open(flag_path, 'r', encoding='utf-8', errors='ignore') as f:
                    _ = f.read()
            except Exception:
                pass
            os.remove(flag_path)
            logger.info("재학습 플래그 감지")
            return True
        return False

    def retrain_pipeline(self) -> bool:
        logger.info("=" * 60)
        logger.info("재학습 파이프라인 시작")
        logger.info("=" * 60)
        try:
            # 1) 병합
            logger.info("1. 데이터 병합")
            merged = self.data_merger.merge_all_data()
            if merged is None or merged.empty:
                logger.error("병합 데이터 없음")
                return False

            # 2) 가격 검증
            logger.info("2. 가격 데이터 검증")
            ok, issues = DataValidator.validate_price_data(merged)
            if not ok:
                logger.warning(f"가격 데이터 이슈: {issues}")

            # 3) 모델 재학습
            logger.info("3. 모델 재학습")
            metrics = self.model_optimizer.retrain_model(merged)
            logger.info(f"재학습 결과: {metrics}")

            # 4) 실패 패턴 분석 → 적응형 필터 저장
            logger.info("4. 실패 패턴 분석/필터 저장")
            trade_logs = self.data_merger.load_trade_logs(days=30)
            if not trade_logs.empty:
                patterns = self.model_optimizer.analyze_failures(trade_logs)
                filters_doc = _build_adaptive_filters(patterns)
                apath = os.path.join(self.config.MODEL_DIR, 'adaptive_filters.json')
                with open(apath, 'w', encoding='utf-8') as f:
                    json.dump(filters_doc, f, ensure_ascii=False, indent=2)
                logger.info(f"적응형 필터 저장 완료: {apath} (활성 {len(filters_doc.get('active_filters', []))}개)")
            else:
                logger.info("거래 로그 부족 → 필터 업데이트 생략")

            # 5) 실시간 모듈 새 모델 로드
            if self.real_trader:
                if self.model_trainer.load_model():
                    logger.info("실시간 모듈: 새 모델 적용")
                if hasattr(self.real_trader, "load_adaptive_filters"):
                    self.real_trader.adaptive_filters = self.real_trader.load_adaptive_filters()

            self.last_retrain_time = datetime.now(timezone.utc)
            with open(os.path.join(self.config.BASE_DIR, '.retrain_complete'), 'w', encoding='utf-8') as f:
                f.write(self.last_retrain_time.isoformat())
            logger.info("재학습 완료 플래그 기록")

            logger.info("재학습 파이프라인 완료")
            return True

        except Exception as e:
            logger.exception(f"재학습 파이프라인 오류: {e}")
            return False

    # ── 라이브 루프/스케줄러 ─────────────────────────────────────────────────
    def trading_loop(self):
        logger.info("거래 루프 시작")
        while self.is_running:
            try:
                # 내부에서 주기적 실행/폴링 (논블로킹 재학습)
                self.real_trader.run_live_trading(
                    duration_hours=999999,
                    trade_interval_minutes=self.config.TRADE_INTERVAL_MINUTES
                )

                # (보통 여기 도달하지 않음: 예외/중지시만)
                self.monitor.generate_report()

                if self.check_retrain_flag():
                    self.retrain_pipeline()

            except Exception as e:
                logger.exception(f"거래 루프 오류: {e}")
                time.sleep(60)

    def scheduled_tasks(self):
        # 자정: 청소
        schedule.every().day.at("00:00").do(self.daily_maintenance)
        # 4시간마다: 리포트
        schedule.every(4).hours.do(self.monitor.generate_report)
        # 3시간마다: 자동 재학습 체크
        schedule.every(3).hours.do(self.auto_retrain_check)

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)

    def daily_maintenance(self):
        logger.info("일일 유지보수 시작")
        try:
            self.data_merger.cleanup_old_data(days_to_keep=90)

            # 로그 로테이션
            lf = 'pipeline.log'
            if os.path.exists(lf):
                size_mb = os.path.getsize(lf) / (1024 * 1024)
                if size_mb > 100:
                    archive = f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
                    try:
                        os.rename(lf, archive)
                        logger.info(f"로그 아카이브: {archive}")
                    except Exception:
                        pass

            # 성과 저장
            stats = self.monitor.analyze_recent_trades(30)
            if stats:
                path = os.path.join(self.config.BASE_DIR, f"stats_{datetime.now().strftime('%Y%m%d')}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            logger.exception("일일 유지보수 오류")
        logger.info("일일 유지보수 완료")

    def auto_retrain_check(self):
        logger.info("자동 재학습 검토(윌슨 하한)…")
        try:
            day = self.monitor.analyze_recent_trades(1)
            n = int(day.get('total_trades', 0)) if day else 0
            min_n = self.config.EVALUATION_WINDOW

            if n < min_n:
                logger.info(f"표본 부족 → 보류 (n={n} < {min_n})")
                return

            wins = int(day.get('wins', 0)) if day else 0
            L = wilson_lower_bound(wins, n, z=1.96)
            wr = wins / n if n > 0 else 0.0
            logger.info(f"최근 n={n}, wr={wr:.2%}, 윌슨 하한={L:.2%}, 임계={self.config.RETRAIN_THRESHOLD:.2%}")

            min_gap_hours = 6
            gap_ok = (self.last_retrain_time is None) or (
                datetime.now(timezone.utc) - self.last_retrain_time >= timedelta(hours=min_gap_hours)
            )

            if (L < self.config.RETRAIN_THRESHOLD) and gap_ok:
                logger.info("조건 충족 → 재학습 수행")
                self.retrain_pipeline()
            else:
                if not gap_ok:
                    remain = timedelta(hours=min_gap_hours) - (datetime.now(timezone.utc) - self.last_retrain_time)
                    logger.info(f"재학습 최소 간격 미충족. 남은 시간: {remain}")
                else:
                    logger.info("승률 하한이 임계 이상 → 재학습 스킵")
        except Exception:
            logger.exception("자동 재학습 검토 오류")

    # ── 라이프사이클 ──────────────────────────────────────────────────────────
    def start(self):
        logger.info("=" * 60)
        logger.info("바이너리 옵션 예측 시스템 시작")
        logger.info("=" * 60)

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

        logger.info("서비스 시작 완료")

        try:
            while self.is_running:
                time.sleep(10)
                self.monitor_system_health()
        except KeyboardInterrupt:
            logger.info("종료 신호 수신")
            self.stop()

    def monitor_system_health(self):
        if self.trading_thread and not self.trading_thread.is_alive():
            logger.warning("거래 스레드 중지 감지 → 재시작")
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()

    def stop(self):
        logger.info("시스템 종료 중…")
        self.is_running = False
        try:
            if self.real_trader:
                self.real_trader.is_running = False
                if hasattr(self.real_trader, "print_performance_summary"):
                    self.real_trader.print_performance_summary()
            self.monitor.generate_report()
            if self.data_merger.merged_data is not None:
                self.data_merger.save_merged_data(self.data_merger.merged_data)
        except Exception:
            logger.exception("종료 루틴 오류")
        logger.info("시스템 종료 완료")

    # ── 기타: 백테스트/옵티마이즈 ─────────────────────────────────────────────
    def run_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        logger.info("백테스팅 모드 시작")
        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        hist = self.data_merger.load_price_data(
            start_date=pd.to_datetime(start_date, utc=True) if start_date else None,
            end_date=pd.to_datetime(end_date, utc=True) if end_date else None,
        )
        if hist.empty:
            logger.error("백테스트용 가격 데이터 없음")
            return

        results = self.real_trader.backtest(hist, start_date, end_date)
        if results is not None:
            results.to_csv('backtest_results.csv', index=False)
            logger.info("백테스트 결과 저장: backtest_results.csv")

    def optimize_hyperparameters(self):
        logger.info("하이퍼파라미터 최적화 시작")

        X, y = self.data_merger.get_training_data(lookback_days=90)
        if X is None or y is None:
            logger.error("학습 데이터 준비 실패")
            return

        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.8, 0.9],
            'bagging_fraction': [0.7, 0.8, 0.9],
        }

        best_params = None
        best_score = -1e9

        for nl in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                for ff in param_grid['feature_fraction']:
                    for bf in param_grid['bagging_fraction']:
                        params = self.config.LGBM_PARAMS.copy()
                        params.update({
                            'num_leaves': nl,
                            'learning_rate': lr,
                            'feature_fraction': ff,
                            'bagging_fraction': bf
                        })

                        temp_trainer = ModelTrainer(self.config)
                        temp_trainer.config.LGBM_PARAMS = params

                        try:
                            # 시계열 피처 선택 + 앙상블 학습
                            sel = temp_trainer.feature_selection_temporal(X, y, top_k=50)
                            temp_trainer.selected_features = sel
                            metrics = temp_trainer.train_ensemble_temporal(X, y)
                            val_acc = metrics['validation']['accuracy']
                        except Exception as e:
                            logger.warning(f"파라미터 조합 실패: {e}")
                            continue

                        if val_acc > best_score:
                            best_score = val_acc
                            best_params = params
                            logger.info(f"새 최적 파라미터: acc={val_acc:.4f} / {best_params}")

        if best_params:
            self.config.LGBM_PARAMS = best_params
            if hasattr(self.config, 'save_config'):
                try:
                    self.config.save_config()
                except Exception:
                    pass
            logger.info(f"최적 파라미터: {best_params} | 최고 검증 정확도: {best_score:.4f}")
        else:
            logger.info("더 나은 파라미터를 찾지 못함")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='바이너리 옵션 예측 시스템')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'optimize'], default='live')
    parser.add_argument('--start-date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, help='설정 파일 경로(JSON)')

    args = parser.parse_args()

    # 커스텀 설정 적용
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            custom = json.load(f)
        for k, v in custom.items():
            if hasattr(Config, k):
                setattr(Config, k, v)

    pipe = MainPipeline()

    if args.mode == 'live':
        pipe.start()

    elif args.mode == 'backtest':
        pipe.run_backtest(args.start_date, args.end_date)

    elif args.mode == 'train':
        if not pipe.initialize_system():
            logger.error("시스템 초기화 실패")
            return
        data = pipe.fetch_initial_training_data()
        if data is not None and not data.empty:
            metrics = pipe.model_optimizer.initial_training(data)
            logger.info("학습 완료")
            for split, metric in metrics.items():
                logger.info(f"{split}: {metric}")

    elif args.mode == 'optimize':
        pipe.optimize_hyperparameters()


if __name__ == "__main__":
    main()
