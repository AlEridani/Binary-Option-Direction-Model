# main_pipe.py - 메인 자동화 파이프라인 (논블로킹 재학습/적응형 필터 호환)

import os
import sys
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from pathlib import Path

# 모듈 임포트
from config import Config
from model_train import ModelTrainer, ModelOptimizer, FeatureEngineer
from data_merge import DataMerger, DataValidator
from real_trade import RealTimeTrader, BinanceAPIClient, TradingMonitor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _to_utc_series(s):
    # 어떤 형식이 와도 UTC-aware로 강제
    s = pd.to_datetime(s, utc=True, errors='coerce')
    return s

def _coerce_to_utc(series: pd.Series) -> pd.Series:
    """
    어떤 형태(문자열/naive/aware/혼합)로 와도 UTC-aware(datetime64[ns, UTC])로 강제 변환.
    """
    # 시리즈 전체를 한 번에 변환 시도
    s = pd.to_datetime(series, errors='coerce', utc=False)

    # 1) 이미 UTC-aware면 그대로
    if getattr(s.dt, "tz", None) is not None:
        try:
            return s.dt.tz_convert("UTC")
        except Exception:
            pass  # 혼합/오브젝트일 수 있으니 아래에서 행 단위 처리

    # 2) tz-naive datetime64[ns] 인 경우
    if np.issubdtype(getattr(s, "dtype", None), np.datetime64) and getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")

    # 3) 혼합(object)일 수 있으니 행 단위 보정
    def _to_utc_one(x):
        if pd.isna(x):
            return pd.NaT
        dt = pd.to_datetime(x, errors='coerce')
        if pd.isna(dt):
            return pd.NaT
        if getattr(dt, "tzinfo", None) is None:
            return dt.tz_localize("UTC")
        return dt.tz_convert("UTC")

    fixed = series.apply(_to_utc_one)
    # 최종 dtype을 강제로 통일
    return fixed.astype("datetime64[ns, UTC]")

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
    model_train.ModelOptimizer.analyze_failures() 출력 -> 실시간 필터 스키마 변환
    RealTimeTrader.load_adaptive_filters()가 기대하는 구조:
      {
        "active_filters": [
          {
            "name": "...",
            "type": "risk" | "volume" | "time" | "rsi" | ...,
            "field": "atr_14"|"volume_ratio"|"hour"|"rsi_14",
            "operator": ">"|"<"|"in"|"extreme"|"between",
            "threshold": float,
            "bad_hours": [int, ...],
            "lower_threshold": float,
            "upper_threshold": float,
            "condition": "human readable",
            "reason": "...",
            "improvement": 0.0
          },
          ...
        ],
        "filter_history": [...]
      }
    """
    active = []

    if not failure_patterns:
        return {"active_filters": [], "filter_history": []}

    # 1) 변동성
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
            "reason": fp.get("reason", "고변동성 구간에서 실패율↑"),
            "improvement": 0.0
        })

    # 2) 거래량
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
            "reason": fp.get("reason", "저거래량/유동성 부족 구간 실패율↑"),
            "improvement": 0.0
        })

    # 3) 시간대
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

    # 4) RSI 극단값
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
            "reason": fp.get("reason", "RSI 극단값에서 실패율↑"),
            "improvement": 0.0
        })

    return {"active_filters": active, "filter_history": []}


class MainPipeline:
    """메인 자동화 파이프라인"""

    def __init__(self):
        # 설정 초기화
        Config.create_directories()
        self.config = Config

        # 컴포넌트 초기화
        self.model_trainer = ModelTrainer(Config)
        self.model_optimizer = ModelOptimizer(Config)
        self.data_merger = DataMerger(Config)
        self.api_client = BinanceAPIClient()
        self.real_trader = None
        self.monitor = TradingMonitor(Config)

        # 상태 변수
        # is_running은 run_live_trading 내부 루프와 별개로 파이프라인 생명주기 제어
        self.is_running = False
        self.last_retrain_time = None
        self.trading_thread = None

        logger.info("메인 파이프라인 초기화 완료")

    def initialize_system(self):
        logger.info("시스템 초기화 시작...")

        # 1) 모델 확인/학습 or 로드
        model_path = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        if not os.path.exists(model_path):
            logger.info("학습된 모델이 없습니다. 초기 학습을 시작합니다.")
            initial_data = self.fetch_initial_training_data()
            if initial_data is not None and not initial_data.empty:
                metrics = self.model_optimizer.initial_training(initial_data)
                logger.info(f"초기 학습 완료. 테스트 승률: {metrics['test']['win_rate']:.2%}")
            else:
                logger.error("초기 학습용 데이터를 가져올 수 없습니다.")
                return False
        else:
            if self.model_trainer.load_model():
                logger.info("기존 모델 로드 완료")
            else:
                logger.error("모델 로드 실패")
                return False

        # 2) 실시간 거래 객체
        self.real_trader = RealTimeTrader(
            config=self.config,
            model_trainer=self.model_trainer,
            api_client=self.api_client
        )
        return True

    def fetch_initial_training_data(self):
        logger.info("초기 학습 데이터 수집 중...")
        try:
            all_data = []
            for _ in range(1):  # 실제 운용 시 구간 분할 반복 수 늘리세요
                df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.1)
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                self.save_price_data(combined)
                logger.info(f"초기 데이터 수집 완료: {len(combined)} 레코드")
                return combined
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")

        logger.info("시뮬레이션 데이터 생성 중...")
        return self.generate_simulation_data(days=270)

    def generate_simulation_data(self, days=270):
        periods = days * 24 * 60
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='1min')

        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, periods)
        price = 42000 * np.exp(np.cumsum(returns))

        data = []
        for i, ts in enumerate(timestamps):
            base = price[i]
            o = base * (1 + np.random.uniform(-0.001, 0.001))
            c = base * (1 + np.random.uniform(-0.001, 0.001))
            h = max(o, c) * (1 + np.random.uniform(0, 0.002))
            l = min(o, c) * (1 - np.random.uniform(0, 0.002))
            v = np.random.uniform(100, 1000) * (1 + 0.5 * np.sin(i/1440 * 2 * np.pi))
            data.append({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

        df = pd.DataFrame(data)
        self.save_price_data(df)
        return df

    def save_price_data(self, df):
        os.makedirs(self.config.PRICE_DATA_DIR, exist_ok=True)
        path = os.path.join(self.config.PRICE_DATA_DIR, 'prices.csv')

        # 새로 들어온 DF도 UTC 강제
        if 'timestamp' not in df.columns:
            raise ValueError("입력 df에 'timestamp' 컬럼이 없습니다.")
        df = df.copy()
        df['timestamp'] = _coerce_to_utc(df['timestamp'])
        df['ts_min'] = df['timestamp'].dt.floor('T')

        if os.path.exists(path):
            # 과거 파일 로드 후 UTC 강제
            old = pd.read_csv(path)
            if 'timestamp' not in old.columns:
                raise ValueError("기존 prices.csv에 'timestamp' 컬럼이 없습니다.")
            old['timestamp'] = _coerce_to_utc(old['timestamp'])
            if 'ts_min' not in old.columns:
                old['ts_min'] = old['timestamp'].dt.floor('T')

            combined = pd.concat([old, df], ignore_index=True, copy=False)
        else:
            combined = df

        # 이제 모두 UTC-aware → 안전하게 정렬/중복제거
        combined = (
            combined
            .sort_values('timestamp')
            .drop_duplicates(subset=['ts_min'], keep='last')
        )

        combined.to_csv(path, index=False, encoding='utf-8-sig')

    def fetch_live_data(self):
        try:
            df = self.api_client.get_klines(symbol="BTCUSDT", interval="1m", limit=500)
            if not df.empty:
                self.data_merger.add_new_price_data(df.tail(10))
                return df
        except Exception as e:
            logger.error(f"실시간 데이터 수집 오류: {e}")
        return None

    def check_retrain_flag(self):
        flag_path = os.path.join(self.config.BASE_DIR, '.retrain_required')
        if os.path.exists(flag_path):
            with open(flag_path, 'r', encoding='utf-8', errors='ignore') as f:
                flag_time = f.read()
            os.remove(flag_path)
            logger.info(f"재학습 플래그 감지: {flag_time}")
            return True
        return False

    def retrain_pipeline(self):
        logger.info("="*60)
        logger.info("재학습 파이프라인 시작")
        logger.info("="*60)

        try:
            # 1) 데이터 병합
            logger.info("1. 데이터 병합 중...")
            merged_data = self.data_merger.merge_all_data()
            if merged_data is None or merged_data.empty:
                logger.error("병합할 데이터가 없습니다.")
                return False

            # 2) 데이터 검증(가격 측면)
            logger.info("2. 데이터 검증 중...")
            is_valid, issues = DataValidator.validate_price_data(merged_data)
            if not is_valid:
                logger.warning(f"데이터 이슈: {issues}")

            # 3) 모델 재학습
            logger.info("3. 모델 재학습 중...")
            metrics = self.model_optimizer.retrain_model(merged_data)
            logger.info(f"재학습 결과: train={metrics['train']}, val={metrics['validation']}, test={metrics['test']}")

            # 4) 실패 패턴 분석 -> 적응형 필터 저장 (MODEL_DIR/adaptive_filters.json)
            logger.info("4. 실패 패턴 분석 및 필터 저장...")
            trade_logs = self.data_merger.load_trade_logs()
            if not trade_logs.empty:
                failure_patterns = self.model_optimizer.analyze_failures(trade_logs)
                filters_doc = _build_adaptive_filters(failure_patterns)
                path = os.path.join(self.config.MODEL_DIR, 'adaptive_filters.json')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(filters_doc, f, ensure_ascii=False, indent=2)
                logger.info(f"적응형 필터 저장 완료: {path} (활성 {len(filters_doc.get('active_filters', []))}개)")
            else:
                logger.info("거래 로그가 없어 필터 업데이트 생략")

            # 5) 실시간 모듈에 새 모델 적용 (파일 기반이므로 load_model만)
            if self.real_trader:
                if self.real_trader.model_trainer.load_model():
                    logger.info("실시간 거래 모듈: 새 모델 로드 완료")
                self.real_trader.adaptive_filters = self.real_trader.load_adaptive_filters()

            # 완료 플래그 생성 (논블로킹 루프가 이걸 감지)
            self.last_retrain_time = datetime.now(timezone.utc)
            complete_flag = os.path.join(self.config.BASE_DIR, '.retrain_complete')
            with open(complete_flag, 'w', encoding='utf-8') as f:
                f.write(datetime.now(timezone.utc).isoformat())
            logger.info("재학습 완료 플래그 생성: .retrain_complete")

            logger.info("재학습 파이프라인 완료")
            return True

        except Exception as e:
            logger.exception(f"재학습 파이프라인 오류: {e}")
            return False

    def trading_loop(self):
        logger.info("거래 루프 시작")
        while self.is_running:
            try:
                # run_live_trading는 내부 루프를 돌며 논블로킹 재학습 플래그를 폴링함
                self.real_trader.run_live_trading(
                    duration_hours=99999,
                    trade_interval_minutes=self.config.TRADE_INTERVAL_MINUTES
                )

                # (루프가 끝나는 경우는 거의 없음. 예외/중지 시 아래 로직 실행)
                self.monitor.generate_report()

                # 혹시 외부에서 플래그만 만들어진 경우 처리
                if self.check_retrain_flag():
                    self.retrain_pipeline()

            except Exception as e:
                logger.exception(f"거래 루프 오류: {e}")
                time.sleep(60)

    def scheduled_tasks(self):
        # 매일 자정: 데이터 정리
        schedule.every().day.at("00:00").do(self.daily_maintenance)
        # 매 4시간: 리포트
        schedule.every(4).hours.do(self.monitor.generate_report)
        # 매 3시간: 자동 재학습 검토
        schedule.every(3).hours.do(self.auto_retrain_check)

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)

    def daily_maintenance(self):
        logger.info("일일 유지보수 시작")
        self.data_merger.cleanup_old_data(days_to_keep=90)

        # 로그 로테이션
        log_file = 'pipeline.log'
        if os.path.exists(log_file):
            size_mb = os.path.getsize(log_file) / (1024 * 1024)
            if size_mb > 100:
                archive = f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
                os.rename(log_file, archive)
                logger.info(f"로그 아카이브: {archive}")

        # 성과 저장
        stats = self.monitor.analyze_recent_trades(30)
        if stats:
            stats_path = os.path.join(self.config.BASE_DIR, f"stats_{datetime.now().strftime('%Y%m%d')}.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
        logger.info("일일 유지보수 완료")

    def auto_retrain_check(self):
        logger.info("자동 재학습 검토(윌슨 하한) 중...")

        day_stats = self.monitor.analyze_recent_trades(1)
        n = int(day_stats.get('total_trades', 0)) if day_stats else 0
        min_n = self.config.EVALUATION_WINDOW

        if n < min_n:
            logger.info(f"표본 부족으로 재학습 보류: n={n} < {min_n}")
            return

        wins = int(day_stats.get('wins', 0)) if day_stats else 0
        L = wilson_lower_bound(wins, n, z=1.96)
        wr = wins / n if n > 0 else 0.0
        logger.info(f"최근 표본 n={n}, 승률={wr:.2%}, 윌슨 하한={L:.2%}, 임계={self.config.RETRAIN_THRESHOLD:.2%}")

        min_gap_hours = 6
        gap_ok = (self.last_retrain_time is None) or (
            datetime.now(timezone.utc) - self.last_retrain_time >= timedelta(hours=min_gap_hours)
        )

        if (L < self.config.RETRAIN_THRESHOLD) and gap_ok:
            logger.info("조건 충족 → 재학습 파이프라인 실행")
            self.retrain_pipeline()
        else:
            if not gap_ok:
                remain = timedelta(hours=min_gap_hours) - (datetime.now(timezone.utc) - self.last_retrain_time)
                logger.info(f"재학습 최소 간격 미충족. 남은 시간: {remain}")
            else:
                logger.info("승률 하한이 임계 이상 → 재학습 스킵")

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
        if self.data_merger.merged_data is not None:
            self.data_merger.save_merged_data()
        logger.info("시스템 종료 완료")

    def run_backtest(self, start_date=None, end_date=None):
        logger.info("백테스팅 모드 시작")
        if not self.initialize_system():
            logger.error("시스템 초기화 실패")
            return

        historical_data = self.data_merger.load_price_data(start_date, end_date)
        if historical_data.empty:
            logger.error("백테스팅용 데이터가 없습니다.")
            return

        results = self.real_trader.backtest(historical_data, start_date, end_date)
        if results is not None:
            results.to_csv('backtest_results.csv', index=False)
            logger.info("백테스팅 결과 저장: backtest_results.csv")

    def optimize_hyperparameters(self):
        logger.info("하이퍼파라미터 최적화 시작")

        X, y = self.data_merger.get_training_data(lookback_days=90)
        if X is None:
            logger.error("학습 데이터를 로드할 수 없습니다.")
            return

        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.8, 0.9],
            'bagging_fraction': [0.7, 0.8, 0.9],
        }

        best_params = None
        best_score = -1e9

        for num_leaves in param_grid['num_leaves']:
            for learning_rate in param_grid['learning_rate']:
                for feature_fraction in param_grid['feature_fraction']:
                    for bagging_fraction in param_grid['bagging_fraction']:
                        params = self.config.LGBM_PARAMS.copy()
                        params.update({
                            'num_leaves': num_leaves,
                            'learning_rate': learning_rate,
                            'feature_fraction': feature_fraction,
                            'bagging_fraction': bagging_fraction
                        })

                        temp_trainer = ModelTrainer(self.config)
                        temp_trainer.config.LGBM_PARAMS = params

                        # 시계열 피처 선택 + 앙상블 학습
                        try:
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
            logger.info("더 나은 파라미터를 찾지 못했습니다.")


# CLI
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

    # 커스텀 설정 적용 (있으면)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        for key, value in custom_config.items():
            if hasattr(Config, key):
                setattr(Config, key, value)

    if args.mode == 'live':
        pipeline.start()

    elif args.mode == 'backtest':
        start_date = pd.to_datetime(args.start_date, utc=True) if args.start_date else None
        end_date = pd.to_datetime(args.end_date, utc=True) if args.end_date else None
        pipeline.run_backtest(start_date, end_date)

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
        pipeline.optimize_hyperparameters()



if __name__ == "__main__":
    main()
